//! nullforge-keygen: one-time setup for encrypted key output.
//!
//! Run this on a TRUSTED machine. It:
//!   * writes the PUBLIC age recipient to `age-recipient.txt` (all the miner
//!     needs — safe to keep on the mining box), and
//!   * prints the SECRET identity to STDOUT — this is the ONLY thing that can
//!     decrypt mined keys, so store it OFFLINE (password manager / hardware /
//!     paper) and do NOT leave it on the mining box.
//!
//! Typical use, saving the secret straight to removable/offline media:
//!   nullforge-keygen > /Volumes/OFFLINE/age-identity.txt
use age::secrecy::ExposeSecret;

fn main() {
    let identity = age::x25519::Identity::generate();
    let recipient = identity.to_public();
    let recipient_str = recipient.to_string(); // age1...

    let path = "age-recipient.txt";
    if let Err(e) = std::fs::write(path, format!("{recipient_str}\n")) {
        eprintln!("ERROR: could not write {path}: {e}");
        std::process::exit(1);
    }

    // Human-facing guidance on stderr; the secret itself on stdout so it can be
    // redirected to offline media without the warnings getting mixed in.
    eprintln!("=====================================================================");
    eprintln!(" nullforge age identity generated.");
    eprintln!();
    eprintln!(" PUBLIC recipient  ->  {path}");
    eprintln!("   {recipient_str}");
    eprintln!("   The miner encrypts to this. Safe to keep on the mining box.");
    eprintln!();
    eprintln!(" SECRET identity   ->  printed to STDOUT below. STORE IT OFFLINE NOW:");
    eprintln!("   * It is the ONLY key that can decrypt mined private keys.");
    eprintln!("   * Do NOT leave it on the mining box. Save to a password manager /");
    eprintln!("     hardware / paper, then clear your terminal scrollback.");
    eprintln!("   * Decrypt later on an offline box with:  nullforge-decrypt -i <file>");
    eprintln!("=====================================================================");

    println!("{}", identity.to_string().expose_secret());
}
