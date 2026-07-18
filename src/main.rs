use std::env;
use ethers::signers::{Signer, Wallet};
use ethers::utils::{hex, secret_key_to_address};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::task;
use std::time::{Duration, Instant};
use ethers::core::k256::ecdsa::SigningKey;
use zerohunt::erc8117;

/// Iterations each worker accumulates before flushing to the shared
/// generated-counter atomic (reduces cross-thread cache-line contention).
const COUNTER_FLUSH: usize = 4096;

#[tokio::main]
async fn main() {
    // First CLI arg is the target leading-zero count; defaults to 8 when omitted.
    // On a non-numeric arg, exit cleanly with usage instead of panicking.
    let max_zeros: usize = match env::args().nth(1) {
        None => 8,
        Some(arg) => match arg.trim().parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!(
                    "Invalid leading-zero count: {arg:?}\nUsage: zerohunt [max_zeros]   (positive integer, default 8)"
                );
                std::process::exit(2);
            }
        },
    };
    let num_threads = num_cpus::get();
    println!("Number of threads: {}\nfinding first wallet with {} leading zeros", num_threads, max_zeros);
    let max_zero_count = Arc::new(AtomicUsize::new(0));
    let max_order_chars = Arc::new(AtomicUsize::new(0));
    let best_wallet = Arc::new(Mutex::new(None));

    let mut handles = Vec::new();

    let start_time = Instant::now();
    let total_generated = Arc::new(AtomicUsize::new(0));

    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_clone = Arc::clone(&stop_signal);
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        stop_signal_clone.store(true, Ordering::SeqCst);
        println!("Received Ctrl+C. Stopping...");
    });

    let file = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("scanned_keys.txt")
            .expect("Unable to open file"),
    ));

    for _ in 0..num_threads {
        let max_zero_count = Arc::clone(&max_zero_count);
        let max_order_chars = Arc::clone(&max_order_chars);
        let best_wallet = Arc::clone(&best_wallet);
        let total_generated = Arc::clone(&total_generated);
        let stop_signal = Arc::clone(&stop_signal);
        let file = Arc::clone(&file);

        let handle = task::spawn_blocking(move || {
            // Per-thread CSPRNG: seed ONCE from full OS entropy (the profanity-safe
            // requirement — full-entropy seed, not a weak counter), then stream keys
            // from a fast userspace ChaCha CSPRNG instead of hitting the OS entropy
            // source on every key. Same cryptographic quality, far less overhead.
            let mut rng = StdRng::from_entropy();
            // Accumulate the generated count locally and flush to the shared atomic
            // in batches — a per-iteration atomic add across every thread bounces one
            // cache line and would dominate the hot loop.
            let mut local_generated: usize = 0;

            loop {
                if stop_signal.load(Ordering::Relaxed) {
                    break;
                }

                let signer = SigningKey::random(&mut rng);
                let address = secret_key_to_address(&signer);
                let address_bytes = address.as_bytes();

                // Cheap raw-byte leading-zero-nibble count — the only work on the
                // hot path (no allocation, no formatting).
                let mut zero_count = 0usize;
                for &byte in address_bytes {
                    if byte == 0 {
                        zero_count += 2;
                    } else {
                        zero_count += (byte.leading_zeros() / 4) as usize;
                        break;
                    }
                }

                local_generated += 1;
                if local_generated >= COUNTER_FLUSH {
                    total_generated.fetch_add(local_generated, Ordering::Relaxed);
                    local_generated = 0;
                }

                // Gate ALL expensive work (string format, repeat-char scan, locks,
                // file IO) behind cheap integer checks that are almost always false:
                // below the "interesting" floor of 3, or below the current best.
                // Output-equivalent to the original (nothing < 3 was ever saved; the
                // original merely churned max_zero_count on sub-3 values).
                let current_max = max_zero_count.load(Ordering::Relaxed);
                if zero_count < 3 || zero_count < current_max {
                    continue;
                }

                let address_str = format!("{:?}", address);
                // Invariant: the raw-byte leading-zero-nibble count (`zero_count`)
                // and the hex-string count ERC-8117 uses must never diverge --
                // they are two independent code paths over the same address.
                debug_assert_eq!(
                    zero_count,
                    erc8117::count_leading_zeros(&address_str),
                    "byte-count and hex-string zero-nibble count disagree for {address_str}"
                );
                let chars_in_order = address_str
                    .chars()
                    .skip(zero_count + 2)
                    .fold((None, 0, 0), |(prev_char, max_count, current_count), c| {
                        if Some(c) == prev_char {
                            (prev_char, max_count.max(current_count + 1), current_count + 1)
                        } else {
                            (Some(c), max_count.max(current_count), 1)
                        }
                    })
                    .1;

                let max_order_value = max_order_chars.load(Ordering::Relaxed);
                // Same zero count as the best but not more repeating chars → skip.
                if zero_count == current_max && chars_in_order < max_order_value {
                    continue;
                }

                max_zero_count.store(zero_count, Ordering::SeqCst);
                if chars_in_order > max_order_value {
                    max_order_chars.store(chars_in_order, Ordering::SeqCst);
                }

                let wallet = Wallet::new_with_signer(signer, address, 1);
                let private_key = hex::encode(wallet.signer().to_bytes());

                // Flush the local counter before the (rare) report so the logged and
                // printed running total is current.
                total_generated.fetch_add(local_generated, Ordering::Relaxed);
                local_generated = 0;
                let generated_total = total_generated.load(Ordering::Relaxed);

                {
                    let mut file = file.lock().unwrap();
                    // ERC-8117 subscript form, NON-truncated -> lossless and
                    // reversible (0x0<sub-n> + full remainder reconstructs the
                    // address), so this column stays machine-recoverable on its
                    // own. Sub-threshold hits (n < 4) pass through as raw hex.
                    let address_notated =
                        erc8117::format_address(&address_str, erc8117::Mode::Subscript, false);
                    writeln!(
                        file,
                        "{}\t{}\t{}\t{}",
                        generated_total, address_notated, zero_count, private_key
                    )
                    .expect("Unable to write data to file");
                }

                {
                    let mut best_wallet_lock = best_wallet.lock().unwrap();
                    *best_wallet_lock = Some(wallet);
                }

                // ERC-8117 both-modes, truncated (first4…last4) -- surfaces the
                // high-entropy suffix per the anti-poisoning intent.
                println!(
                    "New best address with {} leading zeros and {} repeating characters: {}",
                    zero_count, chars_in_order, erc8117::format_both(&address_str, true)
                );

                if zero_count >= max_zeros {
                    break;
                }
            }

            // Flush any unreported local count on exit.
            total_generated.fetch_add(local_generated, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    let rate_handle = {
        let total_generated = Arc::clone(&total_generated);
        let start_time = start_time.clone();

        task::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(20)).await;
                let count = total_generated.load(Ordering::Relaxed);
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = count as f64 / elapsed.max(1.0);

                println!("Wallet generation rate: {:.2} wallets/sec", rate);
            }
        })
    };

    for handle in handles {
        let _ = handle.await;
    }

    rate_handle.abort();

    let best_wallet_lock = best_wallet.lock().unwrap();
    if let Some(wallet) = &*best_wallet_lock {
        let addr_str = format!("{:?}", wallet.address());
        println!("Found wallet with the most leading zeros:");
        // Raw hex for copy/paste + the ERC-8117 forms (non-truncated -> the full
        // address is preserved, just with the leading-zero run compacted).
        println!("Address (raw):      {}", addr_str);
        println!("Address (ERC-8117): {}", erc8117::format_both(&addr_str, false));
        println!("Private Key: {}", hex::encode(wallet.signer().to_bytes()));
    } else {
        println!("No wallet found.");
    }
}