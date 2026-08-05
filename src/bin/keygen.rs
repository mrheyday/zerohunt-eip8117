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

#[cfg(test)]
mod tests {
    use age::secrecy::ExposeSecret;
    use std::str::FromStr;

    // These tests exercise the exact `age` API surface `main()` relies on
    // (generate -> to_public -> to_string / expose_secret) without touching
    // the filesystem or stdout, since `main()` itself is not decomposed into
    // a separately callable, side-effect-free function.

    #[test]
    fn generated_recipient_has_age1_prefix_and_round_trips_through_parsing() {
        let identity = age::x25519::Identity::generate();
        let recipient_str = identity.to_public().to_string();
        assert!(recipient_str.starts_with("age1"), "got: {recipient_str}");

        let reparsed = age::x25519::Recipient::from_str(&recipient_str).unwrap();
        assert_eq!(reparsed.to_string(), recipient_str);
    }

    #[test]
    fn generated_secret_has_the_expected_age_identity_prefix() {
        let identity = age::x25519::Identity::generate();
        let secret_str = identity.to_string().expose_secret().to_string();
        assert!(
            secret_str.to_uppercase().starts_with("AGE-SECRET-KEY-1"),
            "got: {secret_str}"
        );
    }

    #[test]
    fn generated_secret_round_trips_via_from_str_to_the_same_recipient() {
        let identity = age::x25519::Identity::generate();
        let secret_str = identity.to_string().expose_secret().to_string();

        let reloaded = age::x25519::Identity::from_str(&secret_str).unwrap();
        assert_eq!(
            reloaded.to_public().to_string(),
            identity.to_public().to_string()
        );
    }

    #[test]
    fn successive_generate_calls_produce_distinct_identities() {
        let a = age::x25519::Identity::generate();
        let b = age::x25519::Identity::generate();
        assert_ne!(
            a.to_string().expose_secret().to_string(),
            b.to_string().expose_secret().to_string(),
            "two freshly generated identities must not collide"
        );
        assert_ne!(a.to_public().to_string(), b.to_public().to_string());
    }
}
