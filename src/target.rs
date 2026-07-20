//! Leading-zero target validation + feasibility guidance.
//!
//! An EVM address is 20 bytes = 40 hex nibbles, so at most 40 leading-zero
//! nibbles are structurally possible. The miner has no internal cap (the Metal
//! kernel counts as a `uint`, the hit record stores the count in a byte, the
//! ERC-8117 formatter renders multi-digit counts), so targets up to 40 — well
//! above the commonly requested 32 — are supported. What was missing was
//! *validation*: `0` and absurd values (`999`) were silently accepted and mined
//! forever. This module bounds the target and warns when it is astronomically
//! infeasible.

/// Maximum leading-zero nibbles an EVM address can have (20 bytes × 2).
pub const MAX_LEADING_ZERO_NIBBLES: usize = 40;

/// Validate a requested leading-zero target, returning it on success or a
/// human-readable error. Accepts `1..=40`.
pub fn validate_target(n: usize) -> Result<usize, String> {
    if n == 0 {
        return Err("target must be at least 1 leading-zero nibble".to_string());
    }
    if n > MAX_LEADING_ZERO_NIBBLES {
        return Err(format!(
            "target {n} exceeds {MAX_LEADING_ZERO_NIBBLES} — an EVM address has only \
             {MAX_LEADING_ZERO_NIBBLES} hex nibbles"
        ));
    }
    Ok(n)
}

/// A note on how hard a target is: an address with `n` leading-zero nibbles
/// occurs roughly once per `16^n` random keys. Returns a warning for targets
/// where the search is practically hopeless, so the operator is not surprised
/// when the miner runs until Ctrl-C. Kept integer-only (no float) — just a
/// human hint.
pub fn feasibility_note(n: usize) -> Option<String> {
    if n < 12 {
        return None;
    }
    Some(format!(
        "note: an address with {n} leading zeros occurs about once per 16^{n} keys — \
         astronomically infeasible to find in practice; the miner will very likely run \
         until you stop it (Ctrl-C). It still reports every intermediate best."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_supported_range_including_32() {
        assert_eq!(validate_target(1).unwrap(), 1);
        assert_eq!(validate_target(8).unwrap(), 8);
        assert_eq!(validate_target(32).unwrap(), 32);
        assert_eq!(validate_target(MAX_LEADING_ZERO_NIBBLES).unwrap(), 40);
    }

    #[test]
    fn rejects_zero_and_over_max() {
        assert!(validate_target(0).is_err());
        assert!(validate_target(41).is_err());
        assert!(validate_target(999).is_err());
    }

    #[test]
    fn feasibility_note_warns_only_for_high_targets() {
        assert!(feasibility_note(8).is_none());
        assert!(feasibility_note(11).is_none());
        assert!(feasibility_note(12).is_some());
        assert!(feasibility_note(32).unwrap().contains("16^32"));
    }
}
