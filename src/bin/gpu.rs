//! zerohunt-gpu: unified CPU+GPU vanity miner. Mines addresses with the most
//! leading zero nibbles across CPU workers and the Metal GPU against one shared
//! best-tracker, streaming strictly-increasing "new best" records in ERC-8117
//! notation. Stops at `target_zeros` (default 8) or Ctrl-C.
use std::env;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ethers::utils::hex;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tokio::task;

use zerohunt::erc8117;
use zerohunt::gpu::MetalContext;
use zerohunt::miner::cpu::cpu_worker;
use zerohunt::miner::gpu_driver::{run_batches, N_THREADS};
use zerohunt::miner::shared::MinerShared;

/// CPU workers = round(num_cpus * UTILIZATION); ~20% of cores left for the
/// system + the (I/O-bound) GPU driver thread.
const UTILIZATION: f64 = 0.80;

#[tokio::main]
async fn main() {
    // CLI: `zerohunt-gpu [target_zeros]`, default 8 (mirrors the CPU tool).
    let target: usize = match env::args().nth(1) {
        None => 8,
        Some(arg) => match arg.trim().parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!(
                    "Invalid leading-zero count: {arg:?}\nUsage: zerohunt-gpu [target_zeros]   (positive integer, default 8)"
                );
                std::process::exit(2);
            }
        },
    };

    let cpu_workers = ((num_cpus::get() as f64) * UTILIZATION).round().max(1.0) as usize;
    println!(
        "zerohunt-gpu: {cpu_workers} CPU workers + GPU, finding an address with {target} leading zeros"
    );

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("scanned_keys.txt")
        .expect("Unable to open scanned_keys.txt");

    let start = Instant::now();
    let shared = Arc::new(MinerShared::new(target, file, start));

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
            println!("Private Key: {}", hex::encode(best.privkey));
        }
        None => println!("No wallet found."),
    }
}
