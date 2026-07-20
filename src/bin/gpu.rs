//! nullforge-gpu: unified CPU+GPU vanity miner. Mines addresses with the most
//! leading zero nibbles across CPU workers and the Metal GPU against one shared
//! best-tracker, streaming strictly-increasing "new best" records in ERC-8117
//! notation. Stops at `target_zeros` (default 8) or Ctrl-C.
use std::env;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ethers::utils::hex;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tokio::task;

use nullforge::erc8117;
use nullforge::gpu::MetalContext;
use nullforge::miner::cpu::cpu_worker;
use nullforge::miner::gpu_driver::{run_batches, N_THREADS};
use nullforge::miner::shared::MinerShared;

/// CPU workers = round(num_cpus * UTILIZATION); ~20% of cores left for the
/// system + the (I/O-bound) GPU driver thread.
const UTILIZATION: f64 = 0.80;

#[tokio::main]
async fn main() {
    // CLI: `nullforge-gpu [target_zeros] [--reveal]` (EOA mode), or
    //      `nullforge-gpu [target] --create2 --deployer 0x.. --init-code-hash 0x..`
    // Structured parse so flag VALUES (0x..) aren't mistaken for the target.
    let args: Vec<String> = env::args().skip(1).collect();
    let mut reveal = false;
    let mut create2 = false;
    let mut create3 = false;
    let mut deployer_arg: Option<String> = None;
    let mut ich_arg: Option<String> = None;
    let mut factory_arg: Option<String> = None;
    let mut proxyhash_arg: Option<String> = None;
    let mut target_arg: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--reveal" => reveal = true,
            "--create2" => create2 = true,
            "--create3" => create3 = true,
            "--deployer" => {
                i += 1;
                deployer_arg = args.get(i).cloned();
            }
            "--init-code-hash" => {
                i += 1;
                ich_arg = args.get(i).cloned();
            }
            "--factory" => {
                i += 1;
                factory_arg = args.get(i).cloned();
            }
            "--proxy-hash" => {
                i += 1;
                proxyhash_arg = args.get(i).cloned();
            }
            a if a.starts_with("--") => {
                eprintln!(
                    "unknown flag: {a}\nUsage: nullforge-gpu [target] [--reveal] [--create2 --deployer 0x.. --init-code-hash 0x..] [--create3 --factory 0x.. [--proxy-hash 0x..]]"
                );
                std::process::exit(2);
            }
            a if target_arg.is_none() => target_arg = Some(a.to_string()),
            _ => {}
        }
        i += 1;
    }
    let target: usize = match target_arg {
        None => 8,
        Some(arg) => match arg.trim().parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!("Invalid leading-zero count: {arg:?}");
                std::process::exit(2);
            }
        },
    };

    // Validate the target range (1..=40 nibbles) and warn on infeasible targets.
    let target = match nullforge::target::validate_target(target) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(2);
        }
    };
    if let Some(note) = nullforge::target::feasibility_note(target) {
        eprintln!("{note}");
    }

    if create2 && create3 {
        eprintln!("ERROR: --create2 and --create3 are mutually exclusive");
        std::process::exit(2);
    }

    // CREATE2 salt-mining mode: keccak-only; the output is a PUBLIC salt, so no
    // age recipient / encryption is involved. Runs to `target` or Ctrl-C.
    if create2 {
        let deployer: [u8; 20] = parse_hex_arg(deployer_arg, "--deployer", 20)
            .try_into()
            .unwrap();
        let ich: [u8; 32] = parse_hex_arg(ich_arg, "--init-code-hash", 32)
            .try_into()
            .unwrap();

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let stop = Arc::clone(&stop);
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                println!("Received Ctrl+C. Stopping...");
                stop.store(true, std::sync::atomic::Ordering::SeqCst);
            });
        }
        let ctx = MetalContext::new();
        println!("nullforge-gpu --create2: mining a CREATE2 address with {target} leading zeros");
        let handle = task::spawn_blocking(move || {
            nullforge::miner::create2::run_create2(&ctx, &deployer, &ich, target, stop);
        });
        let _ = handle.await;
        return;
    }

    // CREATE3 salt-mining mode: keccak-only, init-code-independent (the deployed
    // address is a function of factory + salt only, via proxy CREATE2 -> proxy
    // CREATE at nonce 1). PUBLIC salt, no encryption. Runs to `target` or Ctrl-C.
    if create3 {
        let factory: [u8; 20] = parse_hex_arg(factory_arg, "--factory", 20)
            .try_into()
            .unwrap();
        // Default to the 0xSequence/Solady proxy hash; only parse --proxy-hash if given.
        let proxy_hash: [u8; 32] = if proxyhash_arg.is_some() {
            parse_hex_arg(proxyhash_arg, "--proxy-hash", 32)
                .try_into()
                .unwrap()
        } else {
            nullforge::miner::create3::DEFAULT_CREATE3_PROXY_HASH
        };

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let stop = Arc::clone(&stop);
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                println!("Received Ctrl+C. Stopping...");
                stop.store(true, std::sync::atomic::Ordering::SeqCst);
            });
        }
        let ctx = MetalContext::new();
        println!(
            "nullforge-gpu --create3: mining a CREATE3 address with {target} leading zeros (proxy 0x{})",
            hex::encode(proxy_hash)
        );
        let handle = task::spawn_blocking(move || {
            nullforge::miner::create3::run_create3(&ctx, &factory, &proxy_hash, target, stop);
        });
        let _ = handle.await;
        return;
    }

    // Resolve how found keys are written. Fail closed if no age recipient is
    // configured and --reveal was not passed.
    let key_sink = match nullforge::keyenc::resolve_sink(reveal) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(2);
        }
    };
    if reveal {
        eprintln!(
            "WARNING: --reveal set -> writing PLAINTEXT private keys to scanned_keys.txt (INSECURE)."
        );
    } else {
        println!(
            "Key output: ENCRYPTED to age recipient (scanned_keys.txt key column is ciphertext)."
        );
    }

    let cpu_workers = ((num_cpus::get() as f64) * UTILIZATION).round().max(1.0) as usize;
    println!(
        "nullforge-gpu: {cpu_workers} CPU workers + GPU, finding an address with {target} leading zeros"
    );

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open("scanned_keys.txt")
        .expect("Unable to open scanned_keys.txt");

    let start = Instant::now();
    let shared = Arc::new(MinerShared::new(target, file, start, key_sink));

    // Build the GPU context up front so a missing device fails fast and clearly.
    let ctx = MetalContext::new();

    // Full-entropy per-thread seeds (generated once; the GPU driver advances the
    // per-thread counter across batches, so keys never repeat).
    let mut rng = StdRng::from_entropy();
    let mut seeds = vec![[0u8; 32]; N_THREADS];
    for s in seeds.iter_mut() {
        rng.fill_bytes(s);
    }

    // Ctrl-C -> request stop.
    {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            println!("Received Ctrl+C. Stopping...");
            shared.request_stop();
        });
    }

    // Rate reporter (per-engine attribution) every 5s until stop.
    let rate_handle = {
        let shared = Arc::clone(&shared);
        task::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                if shared.should_stop() {
                    break;
                }
                let secs = shared.elapsed_secs().max(1.0);
                let cpu = shared.cpu_keys() as f64 / secs;
                let gpu = shared.gpu_keys() as f64 / secs;
                println!(
                    "CPU: {:.2} Mkeys/s | GPU: {:.2} Mkeys/s | total {:.2} Mkeys/s | best: {}",
                    cpu / 1e6,
                    gpu / 1e6,
                    (cpu + gpu) / 1e6,
                    shared.best_zeros()
                );
            }
        })
    };

    // GPU driver on a dedicated blocking thread (owns `ctx`, `seeds`).
    let gpu_handle = {
        let shared = Arc::clone(&shared);
        task::spawn_blocking(move || run_batches(&ctx, shared, &seeds))
    };

    // CPU workers.
    let mut cpu_handles = Vec::new();
    for _ in 0..cpu_workers {
        let shared = Arc::clone(&shared);
        cpu_handles.push(task::spawn_blocking(move || cpu_worker(shared)));
    }

    let _ = gpu_handle.await;
    for h in cpu_handles {
        let _ = h.await;
    }
    rate_handle.abort();

    match shared.take_best() {
        Some(best) => {
            println!("Found wallet with the most leading zeros:");
            println!("Address (raw):      {}", best.address_str);
            println!(
                "Address (ERC-8117): {}",
                erc8117::format_both(&best.address_str, false)
            );
            if reveal {
                println!("Private Key: {}", hex::encode(best.privkey));
            } else {
                println!(
                    "Private Key: [ENCRYPTED to age recipient in scanned_keys.txt; recover offline with `nullforge-decrypt`]"
                );
            }
        }
        None => println!("No wallet found."),
    }
}

/// Parse a required `0x`-prefixed hex CLI arg of exactly `want_bytes` bytes
/// (for `--deployer` / `--init-code-hash` in CREATE2 mode). Exits with a clear
/// message on any problem.
fn parse_hex_arg(val: Option<String>, flag: &str, want_bytes: usize) -> Vec<u8> {
    let val = val.unwrap_or_else(|| {
        eprintln!("ERROR: {flag} <0x...> is required with --create2");
        std::process::exit(2);
    });
    let s = val.strip_prefix("0x").unwrap_or(&val);
    let bytes = hex::decode(s).unwrap_or_else(|e| {
        eprintln!("ERROR: {flag}: invalid hex: {e}");
        std::process::exit(2);
    });
    if bytes.len() != want_bytes {
        eprintln!(
            "ERROR: {flag}: expected {want_bytes} bytes ({} hex chars), got {}",
            want_bytes * 2,
            bytes.len()
        );
        std::process::exit(2);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: only the success paths are exercised here — every error branch of
    // `parse_hex_arg` calls `std::process::exit`, which would tear down the test
    // process itself rather than fail the assertion, so those paths are not
    // unit-testable in-process (they are covered by the CLI usage doc / would
    // need an out-of-process test that spawns the built binary).

    #[test]
    fn parses_0x_prefixed_hex_of_exact_length() {
        // 20-byte deployer example (40 hex chars after 0x).
        let deployer20 = "0x0102030405060708090a0b0c0d0e0f1011121314".to_string();
        let bytes = parse_hex_arg(Some(deployer20), "--deployer", 20);
        assert_eq!(
            bytes,
            vec![
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14
            ]
        );
    }

    #[test]
    fn parses_hex_without_0x_prefix() {
        let ich = "22".repeat(32); // 32 bytes, no "0x" prefix
        let bytes = parse_hex_arg(Some(ich), "--init-code-hash", 32);
        assert_eq!(bytes, vec![0x22u8; 32]);
    }

    #[test]
    fn parses_all_zero_and_all_ff_edge_values() {
        let zero = format!("0x{}", "00".repeat(20));
        assert_eq!(parse_hex_arg(Some(zero), "--deployer", 20), vec![0u8; 20]);

        let max = format!("0x{}", "ff".repeat(32));
        assert_eq!(
            parse_hex_arg(Some(max), "--init-code-hash", 32),
            vec![0xffu8; 32]
        );
    }
}
