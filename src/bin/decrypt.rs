//! nullforge-decrypt: offline recovery of encrypted mined keys.
//!
//! Run this on a TRUSTED / OFFLINE machine that has your SECRET age identity.
//! It reads `scanned_keys.txt` (or a path you pass), decrypts the base64/age
//! key column with your identity, and prints `address <TAB> zeros <TAB> privkey`.
//!
//!   nullforge-decrypt --identity /Volumes/OFFLINE/age-identity.txt [scanned_keys.txt]
use std::str::FromStr;

use ethers::utils::hex;
use nullforge::keyenc;
use zeroize::Zeroize;

fn usage_exit() -> ! {
    eprintln!(
        "usage: nullforge-decrypt --identity <age-identity-file> [scanned_keys.txt]\n\
         \n  The identity file holds your AGE-SECRET-KEY-1... secret; keep it OFFLINE.\n\
         \n  Env fallback for the identity path: NULLFORGE_AGE_IDENTITY_FILE."
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut identity_path: Option<String> = None;
    let mut input_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--identity" | "-i" => {
                i += 1;
                identity_path = args.get(i).cloned().or_else(|| usage_exit());
            }
            "--help" | "-h" => usage_exit(),
            a if a.starts_with('-') => {
                eprintln!("unknown flag: {a}");
                usage_exit();
            }
            a => input_path = Some(a.to_string()),
        }
        i += 1;
    }

    let identity_path = identity_path
        .or_else(|| std::env::var("NULLFORGE_AGE_IDENTITY_FILE").ok())
        .unwrap_or_else(|| usage_exit());
    let input_path = input_path.unwrap_or_else(|| "scanned_keys.txt".to_string());

    // Load the secret identity (first AGE-SECRET-KEY-1... line).
    let identity = age::x25519::Identity::from_str(id_line).unwrap_or_else(|e| {
        eprintln!("ERROR: invalid age identity: {e}");
        std::process::exit(1);
    });
    id_contents.zeroize();

    // Decrypt each line's key column (col 4). Lines written under --reveal hold
    // plaintext hex there and will simply fail to decrypt (reported, skipped).
    let data = std::fs::read_to_string(&input_path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read {input_path}: {e}");
        std::process::exit(1);
    });
    let mut ok = 0usize;
    let mut failed = 0usize;
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            eprintln!("skip (expected 4 tab-separated columns): {line}");
            continue;
        }
        let (addr, zeros, field) = (cols[1], cols[2], cols[3]);
        match keyenc::decrypt_key_field(&identity, field) {
            Ok(mut pt) => {
                println!("{addr}\t{zeros}\t{}", hex::encode(&pt));
                pt.zeroize();
                ok += 1;
            }
            Err(e) => {
                eprintln!("decrypt failed for {addr}: {e}");
                failed += 1;
            }
        }
    }
    eprintln!("decrypted {ok} key(s), {failed} failure(s).");
}
