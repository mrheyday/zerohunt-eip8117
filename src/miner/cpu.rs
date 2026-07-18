//! CPU mining worker: random full-entropy keys -> address -> report.
use std::sync::Arc;

use ethers::core::k256::ecdsa::SigningKey;
use ethers::utils::secret_key_to_address;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::miner::shared::{Engine, MinerShared};

/// Flush the local key counter to the shared atomic every this many keys, to
/// keep the hot loop off the shared cache line (mirrors `src/main.rs`).
const COUNTER_FLUSH: u64 = 4096;

/// Count leading zero nibbles of an address by raw bytes: a fully-zero byte
/// contributes 2, the first non-zero byte contributes `leading_zeros()/4`
/// (1 if its top nibble is zero, else 0), then stop.
pub fn leading_zero_nibbles(addr: &[u8]) -> usize {
    let mut zeros = 0usize;
    for &byte in addr {
        if byte == 0 {
            zeros += 2;
        } else {
            zeros += (byte.leading_zeros() / 4) as usize;
            break;
        }
    }
    zeros
}

/// Mine random keys until `shared.should_stop()`. Reports any candidate that
/// strictly beats the current best through the shared funnel.
pub fn cpu_worker(shared: Arc<MinerShared>) {
    let mut rng = StdRng::from_entropy();
    let mut local: u64 = 0;

    loop {
        if shared.should_stop() {
            break;
        }
        let signer = SigningKey::random(&mut rng);
        let address = secret_key_to_address(&signer);
        let zeros = leading_zero_nibbles(address.as_bytes());

        local += 1;
        if local >= COUNTER_FLUSH {
            shared.add_keys(Engine::Cpu, local);
            local = 0;
        }

        if zeros > shared.best_zeros() {
            let address_str = format!("{:?}", address);
            let privkey: [u8; 32] = signer.to_bytes().into();
            shared.report_hit(Engine::Cpu, privkey, &address_str, zeros);
        }
    }
    shared.add_keys(Engine::Cpu, local);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_leading_zero_nibbles_like_main() {
        // full zero bytes -> +2 each; first non-zero byte -> +(leading_zeros/4)
        assert_eq!(leading_zero_nibbles(&[0x00, 0x00, 0x0a, 0xff]), 5); // 2+2+1
        assert_eq!(leading_zero_nibbles(&[0x0a, 0xff]), 1);
        assert_eq!(leading_zero_nibbles(&[0xff, 0x00]), 0);
        assert_eq!(leading_zero_nibbles(&[0x00, 0x00]), 4);
    }
}
