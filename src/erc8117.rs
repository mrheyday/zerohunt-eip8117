//! ERC-8117 — "Anti-Poisoning Compact EVM Address Format" (Draft, 2025-12-30).
//!
//! A presentation-layer transformation that compresses the low-entropy run of
//! leading zero nibbles in an EVM address into a compact count, surfacing the
//! high-entropy suffix so address-poisoning lookalikes are easier to spot. It
//! is a purely *visual* transformation: it never changes the underlying address
//! bytes (a consumer MUST strip the notation back to full hex before signing).
//!
//! Two display modes, both keeping the literal `0x0` prefix and encoding the
//! total leading-zero-nibble count `n` after it, followed by the remainder of
//! the address (starting at the first non-zero nibble):
//!
//! * [`Mode::Subscript`] — Unicode subscripts, e.g. `0x0₈abcd…1234` (Mode A;
//!   for wallets/explorers/UI that can render Unicode).
//! * [`Mode::Ascii`] — parenthesised count, e.g. `0x0(8)abcd…1234` (Mode B;
//!   the ASCII fallback for logs, terminals, and legacy APIs).
//!
//! Notation is applied only when `n >= LEADING_ZERO_THRESHOLD` (4); below that
//! the address is returned unchanged.
//!
//! This is a direct port of the spec's reference implementation. NOTE: the
//! example *tables* in ERC-8117 contain several transcription errors (e.g. the
//! ERC-4337 EntryPoint is listed with 7 leading zeros but has 8 zero nibbles;
//! the Uniswap V4 row shows `44444c`/`4444c` for a remainder that is actually
//! `4444C…`). This module and its tests follow the reference *algorithm*, which
//! is authoritative — not those tables.

/// Only apply the notation when an address has at least this many leading zero
/// nibbles after the `0x` prefix (ERC-8117 §1 Trigger Condition).
pub const LEADING_ZERO_THRESHOLD: usize = 4;

/// The two ERC-8117 display modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Mode A: Unicode subscript digits (`0x0₈abcd…`). U+2080–U+2089.
    Subscript,
    /// Mode B: ASCII parenthesis fallback (`0x0(8)abcd…`).
    Ascii,
}

/// Count the leading zero *nibbles* after an optional `0x` prefix — the number
/// of `'0'` characters at the start of the address body, up to (not including)
/// the first non-zero nibble. Mirrors the reference `count_leading_zeros`.
///
/// This operates on the hex string, so it is exactly equal to the raw-byte
/// leading-zero-nibble count the miner computes (`byte.leading_zeros() / 4`
/// summed with 2-per-zero-byte); the two are kept in lockstep by a
/// `debug_assert` at the miner's call site.
pub fn count_leading_zeros(address: &str) -> usize {
    let body = address.strip_prefix("0x").unwrap_or(address);
    body.chars().take_while(|&c| c == '0').count()
}

/// Convert a non-negative integer to its Unicode-subscript string (each decimal
/// digit `d` maps to U+2080+d). Mirrors the reference `_to_subscript`.
fn to_subscript(n: usize) -> String {
    n.to_string()
        .bytes()
        .map(|d| char::from_u32(0x2080 + u32::from(d - b'0')).expect("valid subscript codepoint"))
        .collect()
}

/// Format an EVM address using ERC-8117 notation.
///
/// * `address`  — a full `0x`-prefixed EVM address (casing of the remainder is
///   preserved per §3, so pass whatever casing you want displayed).
/// * `mode`     — [`Mode::Subscript`] or [`Mode::Ascii`].
/// * `truncate` — when `true`, abbreviate the remainder to `first4…last4`
///   (using `…`, U+2026) once it exceeds 8 characters. When `false`, the full
///   remainder is kept, making the result a **lossless, reversible** encoding
///   of the original address.
///
/// Returns the address unchanged when it has fewer than
/// [`LEADING_ZERO_THRESHOLD`] leading zero nibbles.
pub fn format_address(address: &str, mode: Mode, truncate: bool) -> String {
    let n = count_leading_zeros(address);
    if n < LEADING_ZERO_THRESHOLD {
        return address.to_string(); // below threshold — no transformation
    }

    let body = address.strip_prefix("0x").unwrap_or(address);
    // `n` counts ASCII '0' chars (1 byte each) in an ASCII hex body, so the
    // byte index `n` is also the char index of the first non-zero nibble.
    let remainder = &body[n..];

    let prefix = match mode {
        Mode::Subscript => format!("0x0{}", to_subscript(n)),
        Mode::Ascii => format!("0x0({n})"),
    };

    let rem_chars: Vec<char> = remainder.chars().collect();
    if !truncate || rem_chars.len() <= 8 {
        return format!("{prefix}{remainder}");
    }

    // Truncated: first 4 … last 4 characters of the remainder.
    let first4: String = rem_chars[..4].iter().collect();
    let last4: String = rem_chars[rem_chars.len() - 4..].iter().collect();
    format!("{prefix}{first4}…{last4}")
}

/// Render both ERC-8117 modes together as `<subscript>  (<ascii>)`, e.g.
/// `0x0₈abcd…1234  (0x0(8)abcd…1234)`. Below the threshold the address is
/// returned once, unchanged (no redundant doubling).
pub fn format_both(address: &str, truncate: bool) -> String {
    if count_leading_zeros(address) < LEADING_ZERO_THRESHOLD {
        return address.to_string();
    }
    let sub = format_address(address, Mode::Subscript, truncate);
    let ascii = format_address(address, Mode::Ascii, truncate);
    format!("{sub}  ({ascii})")
}

#[cfg(test)]
mod tests {
    use super::*;

    // The one internally-consistent full-address vector in the spec:
    // 0x0000...0001 (39 zero nibbles) -> 0x0₃₉1. Golden case.
    const ALL_ZEROS_BUT_ONE: &str = "0x0000000000000000000000000000000000000001";

    #[test]
    fn golden_spec_vector_n39() {
        assert_eq!(count_leading_zeros(ALL_ZEROS_BUT_ONE), 39);
        // Remainder is "1" (len 1 <= 8), so truncate has no effect.
        assert_eq!(
            format_address(ALL_ZEROS_BUT_ONE, Mode::Subscript, false),
            "0x0\u{2083}\u{2089}1" // 0x0₃₉1
        );
        assert_eq!(
            format_address(ALL_ZEROS_BUT_ONE, Mode::Subscript, true),
            "0x0\u{2083}\u{2089}1"
        );
        assert_eq!(
            format_address(ALL_ZEROS_BUT_ONE, Mode::Ascii, false),
            "0x0(39)1"
        );
    }

    #[test]
    fn threshold_edge_n3_unchanged_n4_transformed() {
        // n = 3 (< threshold): returned unchanged in every mode.
        let n3 = "0x000abcdef0123456789012345678901234567890"; // 3 leading zeros
        assert_eq!(count_leading_zeros(n3), 3);
        assert_eq!(format_address(n3, Mode::Subscript, false), n3);
        assert_eq!(format_address(n3, Mode::Ascii, true), n3);
        assert_eq!(format_both(n3, true), n3); // no redundant doubling below threshold

        // n = 4 (== threshold): transformed. remainder = "abcdef...890".
        let n4 = "0x0000abcdef0123456789012345678901234567890";
        assert_eq!(count_leading_zeros(n4), 4);
        assert_eq!(
            format_address(n4, Mode::Subscript, false),
            "0x0\u{2084}abcdef0123456789012345678901234567890"
        );
        assert_eq!(
            format_address(n4, Mode::Ascii, false),
            "0x0(4)abcdef0123456789012345678901234567890"
        );
    }

    #[test]
    fn truncation_boundary_len8_vs_len9() {
        // Remainder exactly 8 chars -> NOT truncated even with truncate=true.
        // 4 zeros + 8 remainder = body len 12.
        let rem8 = "0x0000abcd1234"; // remainder "abcd1234" (8 chars)
        assert_eq!(count_leading_zeros(rem8), 4);
        assert_eq!(
            format_address(rem8, Mode::Ascii, true),
            "0x0(4)abcd1234"
        );

        // Remainder 9 chars -> truncated to first4…last4.
        let rem9 = "0x0000abcd12345"; // remainder "abcd12345" (9 chars)
        assert_eq!(count_leading_zeros(rem9), 4);
        assert_eq!(
            format_address(rem9, Mode::Ascii, true),
            "0x0(4)abcd\u{2026}2345"
        );
        assert_eq!(
            format_address(rem9, Mode::Subscript, true),
            "0x0\u{2084}abcd\u{2026}2345"
        );
        // Non-truncated keeps the whole remainder (lossless).
        assert_eq!(
            format_address(rem9, Mode::Ascii, false),
            "0x0(4)abcd12345"
        );
    }

    #[test]
    fn remainder_casing_preserved() {
        // EIP-55 mixed casing in the remainder must survive verbatim.
        let mixed = "0x00000000aAbBcCdDeEfF00112233445566778899";
        assert_eq!(count_leading_zeros(mixed), 8);
        assert_eq!(
            format_address(mixed, Mode::Ascii, false),
            "0x0(8)aAbBcCdDeEfF00112233445566778899"
        );
    }

    #[test]
    fn real_addresses_via_reference_algorithm() {
        // ERC-4337 EntryPoint: 4 zero bytes = 8 zero nibbles (the spec table's
        // "7" is wrong). remainder = "71727De22E5E9d8BAf0edAc6f37da032".
        let entrypoint = "0x0000000071727De22E5E9d8BAf0edAc6f37da032";
        assert_eq!(count_leading_zeros(entrypoint), 8);
        assert_eq!(
            format_address(entrypoint, Mode::Subscript, false),
            "0x0\u{2088}71727De22E5E9d8BAf0edAc6f37da032"
        );
        assert_eq!(
            format_address(entrypoint, Mode::Ascii, true),
            "0x0(8)7172\u{2026}a032"
        );

        // Uniswap V4 PoolManager: 11 zero nibbles, remainder starts "4444C…"
        // (spec table's "44444c"/"4444c" are both typos).
        let uniswap = "0x000000000004444C5dc75cB358380D2e3dE08A90";
        assert_eq!(count_leading_zeros(uniswap), 11);
        assert_eq!(
            format_address(uniswap, Mode::Subscript, false),
            "0x0\u{2081}\u{2081}4444C5dc75cB358380D2e3dE08A90"
        );
        assert_eq!(
            format_address(uniswap, Mode::Ascii, true),
            "0x0(11)4444\u{2026}8A90"
        );
    }

    #[test]
    fn format_both_shape() {
        let addr = "0x00000000abcd0123456789012345678901234567";
        assert_eq!(count_leading_zeros(addr), 8);
        // <subscript>  (<ascii>), truncated.
        assert_eq!(
            format_both(addr, true),
            "0x0\u{2088}abcd\u{2026}4567  (0x0(8)abcd\u{2026}4567)"
        );
    }
}
