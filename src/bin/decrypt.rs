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

/// Load a secret age identity from a file: the first line starting with
/// `AGE-SECRET-KEY-1` (case-insensitive, matching the age spec's canonical
/// uppercase form). The raw file contents are zeroized before returning.
fn load_identity(path: &str) -> Result<age::x25519::Identity, String> {
    let mut contents =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let id_line = contents
        .lines()
        .map(str::trim)
        .find(|l| l.to_uppercase().starts_with("AGE-SECRET-KEY-1"))
        .map(str::to_string);
    contents.zeroize();

    let mut id_line = id_line.ok_or_else(|| {
        format!("no AGE-SECRET-KEY-1... line found in {path}")
    })?;
    let identity = age::x25519::Identity::from_str(&id_line)
        .map_err(|e| format!("invalid age identity in {path}: {e}"));
    id_line.zeroize();
    identity
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
    let identity = load_identity(&identity_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

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

#[cfg(test)]
mod tests {
    use super::*;
    use age::secrecy::ExposeSecret;

    fn write_identity_file(contents: &str) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), contents).unwrap();
        f
    }

    #[test]
    fn load_identity_reads_a_valid_identity_file() {
        let identity = age::x25519::Identity::generate();
        let secret = identity.to_string().expose_secret().to_string();
        let f = write_identity_file(&secret);

        let loaded = load_identity(f.path().to_str().unwrap()).unwrap();
        // Same key material -> same derived public recipient.
        assert_eq!(
            loaded.to_public().to_string(),
            identity.to_public().to_string()
        );
    }

    #[test]
    fn load_identity_skips_comment_and_blank_lines() {
        // Mirrors the `age-keygen` output shape: comment lines (created-at,
        // public key) precede the actual AGE-SECRET-KEY-1... secret line.
        let identity = age::x25519::Identity::generate();
        let secret = identity.to_string().expose_secret().to_string();
        let contents = format!(
            "# created: 2026-07-17T00:00:00Z\n# public key: {}\n\n{}\n",
            identity.to_public(),
            secret
        );
        let f = write_identity_file(&contents);

        let loaded = load_identity(f.path().to_str().unwrap()).unwrap();
        assert_eq!(
            loaded.to_public().to_string(),
            identity.to_public().to_string()
        );
    }

    #[test]
    fn load_identity_errors_when_no_secret_key_line_present() {
        let f = write_identity_file("# just a comment\nnot a key at all\n");
        let err = load_identity(f.path().to_str().unwrap()).unwrap_err();
        assert!(err.contains("no AGE-SECRET-KEY-1"), "got: {err}");
    }

    #[test]
    fn load_identity_errors_on_malformed_secret_line() {
        let f = write_identity_file("AGE-SECRET-KEY-1NOTVALIDBECH32DATA\n");
        let err = load_identity(f.path().to_str().unwrap()).unwrap_err();
        assert!(err.contains("invalid age identity"), "got: {err}");
    }

    #[test]
    fn load_identity_errors_when_file_is_missing() {
        let err = load_identity("/nonexistent/path/does-not-exist.txt").unwrap_err();
        assert!(err.contains("cannot read"), "got: {err}");
    }
}
