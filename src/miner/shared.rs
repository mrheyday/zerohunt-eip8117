//! Shared state + the single reporting funnel both mining engines call.
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ethers::utils::hex;

use crate::erc8117;

/// Which engine found a candidate (for attribution + rate accounting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Cpu,
    Gpu,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::Cpu => "CPU",
            Engine::Gpu => "GPU",
        }
    }
}

/// The current best find: a private key whose address has `zeros` leading zero
/// nibbles, plus the rendered address string used for display.
#[derive(Debug, Clone)]
pub struct FoundKey {
    pub privkey: [u8; 32],
    pub address_str: String,
    pub zeros: usize,
}

/// State shared by every CPU worker, the GPU driver, and the rate reporter.
pub struct MinerShared {
    pub target: usize,
    best_zeros: AtomicUsize,
    best: Mutex<Option<FoundKey>>,
    file: Mutex<File>,
    cpu_keys: AtomicU64,
    gpu_keys: AtomicU64,
    stop: AtomicBool,
    #[allow(dead_code)] // used by the rate reporter in the binary
    start: Instant,
}

impl MinerShared {
    pub fn new(target: usize, file: File, start: Instant) -> Self {
        Self {
            target,
            best_zeros: AtomicUsize::new(0),
            best: Mutex::new(None),
            file: Mutex::new(file),
            cpu_keys: AtomicU64::new(0),
            gpu_keys: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            start,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn best_zeros(&self) -> usize {
        self.best_zeros.load(Ordering::Relaxed)
    }

    pub fn add_keys(&self, engine: Engine, n: u64) {
        match engine {
            Engine::Cpu => self.cpu_keys.fetch_add(n, Ordering::Relaxed),
            Engine::Gpu => self.gpu_keys.fetch_add(n, Ordering::Relaxed),
        };
    }

    pub fn cpu_keys(&self) -> u64 {
        self.cpu_keys.load(Ordering::Relaxed)
    }

    pub fn gpu_keys(&self) -> u64 {
        self.gpu_keys.load(Ordering::Relaxed)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// The single reporting funnel. Returns `true` iff this became a new best
    /// (strictly more leading zeros than any prior). On a new best it writes the
    /// `scanned_keys.txt` line (ERC-8117 subscript, non-truncated), prints the
    /// "new best" line (ERC-8117 both modes, truncated), and requests stop when
    /// `zeros >= target`.
    pub fn report_hit(
        &self,
        engine: Engine,
        privkey: [u8; 32],
        address_str: &str,
        zeros: usize,
    ) -> bool {
        // Fast path: no lock unless this strictly beats the current best.
        if zeros <= self.best_zeros.load(Ordering::Relaxed) {
            return false;
        }
        let mut best = self.best.lock().unwrap();
        if zeros <= self.best_zeros.load(Ordering::Relaxed) {
            return false; // lost the race to another thread
        }
        self.best_zeros.store(zeros, Ordering::SeqCst);
        *best = Some(FoundKey {
            privkey,
            address_str: address_str.to_string(),
            zeros,
        });

        let total = self.cpu_keys() + self.gpu_keys();
        let notated = erc8117::format_address(address_str, erc8117::Mode::Subscript, false);
        let privhex = hex::encode(privkey);
        {
            let mut file = self.file.lock().unwrap();
            let _ = writeln!(file, "{}\t{}\t{}\t{}", total, notated, zeros, privhex);
        }
        println!(
            "New best [{}] {} leading zeros: {}",
            engine.label(),
            zeros,
            erc8117::format_both(address_str, true)
        );
        if zeros >= self.target {
            self.request_stop();
        }
        true
    }

    pub fn take_best(&self) -> Option<FoundKey> {
        self.best.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::time::Instant;

    fn ctx(target: usize) -> (MinerShared, tempfile::NamedTempFile) {
        let f = tempfile::NamedTempFile::new().unwrap();
        let shared = MinerShared::new(target, f.reopen().unwrap(), Instant::now());
        (shared, f)
    }

    fn read_file(f: &tempfile::NamedTempFile) -> String {
        let mut s = String::new();
        f.reopen().unwrap().read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn first_qualifying_hit_becomes_best_and_writes_file() {
        let (shared, f) = ctx(8);
        let addr = "0x00000000abcd0123456789012345678901234567"; // 8 zeros
        let became = shared.report_hit(Engine::Gpu, [0u8; 32], addr, 8);
        assert!(became);
        assert_eq!(shared.best_zeros(), 8);
        // file column is subscript non-truncated (0x0₈ + full remainder)
        let line = read_file(&f);
        assert!(line.contains("0x0\u{2088}abcd0123456789012345678901234567"), "got: {line}");
        assert!(line.contains("\t8\t"), "zeros column, got: {line}");
    }

    #[test]
    fn strictly_greater_gating_ignores_equal_or_lower() {
        let (shared, _f) = ctx(8);
        let a = "0x00000000abcd0123456789012345678901234567"; // 8
        assert!(shared.report_hit(Engine::Cpu, [0u8; 32], a, 8));
        // equal zeros -> not a new best
        assert!(!shared.report_hit(Engine::Gpu, [1u8; 32], a, 8));
        // lower zeros -> not a new best
        let b = "0x0000abcd012345678901234567890123456789ab"; // 4
        assert!(!shared.report_hit(Engine::Cpu, [2u8; 32], b, 4));
        assert_eq!(shared.best_zeros(), 8);
    }

    #[test]
    fn stop_trips_exactly_at_target() {
        let (shared, _f) = ctx(6);
        assert!(shared.report_hit(Engine::Gpu, [0u8; 32], "0x00000abc0123456789012345678901234567890a", 5));
        assert!(!shared.should_stop(), "5 < target 6");
        assert!(shared.report_hit(Engine::Gpu, [0u8; 32], "0x000000abc123456789012345678901234567890a", 6));
        assert!(shared.should_stop(), "6 >= target 6");
    }

    #[test]
    fn take_best_returns_latest() {
        let (shared, _f) = ctx(8);
        shared.report_hit(Engine::Cpu, [7u8; 32], "0x0000abc0123456789012345678901234567890ab", 4);
        let best = shared.take_best().unwrap();
        assert_eq!(best.zeros, 4);
        assert_eq!(best.privkey, [7u8; 32]);
    }

    #[test]
    fn key_counters_are_per_engine() {
        let (shared, _f) = ctx(8);
        shared.add_keys(Engine::Cpu, 100);
        shared.add_keys(Engine::Gpu, 250);
        shared.add_keys(Engine::Cpu, 5);
        assert_eq!(shared.cpu_keys(), 105);
        assert_eq!(shared.gpu_keys(), 250);
    }
}
