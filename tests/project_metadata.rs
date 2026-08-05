//! Lightweight text-based regression tests for this PR's non-Rust changed
//! files: `Cargo.toml`, `Dockerfile`, `.gitignore`, and `README.MD`.
//!
//! These files carry no executable logic of their own, so rather than skip
//! them entirely, this asserts the specific facts the PR renamed/introduced
//! (the `zerohunt` -> `nullforge` rename, the 4 binary targets, the new
//! encryption dependencies/tooling, and the new `.gitignore` entries) so a
//! future edit can't silently regress them.
//!
//! Deliberately excluded from the "no lingering `zerohunt` references" check
//! below: `docs/plans/2026-07-19-gpu-incremental-ec-miner.md` and
//! `docs/specs/2026-07-19-gpu-incremental-ec-miner-design.md` -- those are
//! out of scope for this PR's file list and are known to still reference the
//! old project name.

use std::fs;

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

#[test]
fn cargo_toml_declares_the_renamed_package_and_lib() {
    let toml = read("Cargo.toml");
    assert!(toml.contains("name = \"nullforge\""), "package/lib name");
    assert!(toml.contains("path = \"src/lib.rs\""));
}

#[test]
fn cargo_toml_declares_all_four_binaries_with_correct_paths() {
    let toml = read("Cargo.toml");
    let bins = [
        ("nullforge", "src/main.rs"),
        ("nullforge-gpu", "src/bin/gpu.rs"),
        ("nullforge-keygen", "src/bin/keygen.rs"),
        ("nullforge-decrypt", "src/bin/decrypt.rs"),
    ];
    for (name, path) in bins {
        assert!(
            toml.contains(&format!("name = \"{name}\"")),
            "missing [[bin]] name = \"{name}\""
        );
        assert!(
            toml.contains(&format!("path = \"{path}\"")),
            "missing [[bin]] path = \"{path}\""
        );
    }
}

#[test]
fn cargo_toml_declares_the_key_encryption_dependencies() {
    let toml = read("Cargo.toml");
    for dep in ["age = \"0.12\"", "zeroize = \"1\"", "base64 = \"0.22\""] {
        assert!(toml.contains(dep), "missing dependency line: {dep}");
    }
    assert!(
        toml.contains("tempfile"),
        "tempfile must remain a dev-dependency for file-based tests"
    );
}

#[test]
fn dockerfile_builds_and_ships_the_renamed_binary() {
    let dockerfile = read("Dockerfile");
    assert!(
        dockerfile.contains("target/release/nullforge /usr/local/bin/nullforge"),
        "Dockerfile must COPY the renamed `nullforge` binary"
    );
    assert!(
        dockerfile.contains("ENTRYPOINT [\"nullforge\"]"),
        "Dockerfile ENTRYPOINT must reference the renamed binary"
    );
    assert!(
        !dockerfile.to_lowercase().contains("zerohunt"),
        "Dockerfile must not reference the old project name"
    );
}

#[test]
fn gitignore_covers_all_generated_key_and_salt_artifacts() {
    let gitignore = read(".gitignore");
    for entry in [
        "scanned_keys.txt",
        "scanned_salts.txt",
        "age-recipient.txt",
        "age-identity*.txt",
    ] {
        assert!(
            gitignore.lines().any(|l| l.trim() == entry),
            "missing .gitignore entry: {entry}"
        );
    }
}

#[test]
fn readme_documents_the_renamed_project_and_new_tooling() {
    let readme = read("README.MD");
    assert!(readme.contains("# Nullforge"), "README title");
    for needle in [
        "nullforge-keygen",
        "nullforge-decrypt",
        "--create2",
        "--reveal",
        "NULLFORGE_AGE_RECIPIENT",
        "NULLFORGE_AGE_IDENTITY_FILE",
    ] {
        assert!(readme.contains(needle), "README missing {needle:?}");
    }
}

#[test]
fn in_scope_project_files_have_no_lingering_zerohunt_references() {
    // Only the files actually touched by this PR are checked here; the
    // 2026-07-19 docs are a separate, out-of-scope PR and still legitimately
    // reference the old name.
    let in_scope_files = [
        "Cargo.toml",
        "Dockerfile",
        ".gitignore",
        "README.MD",
        "docs/plans/2026-07-17-gpu-vanity-miner.md",
        "docs/plans/2026-07-18-unified-cpu-gpu-miner.md",
        "docs/specs/2026-07-17-gpu-vanity-miner-design.md",
        "docs/specs/2026-07-18-unified-cpu-gpu-max-leading-zero-miner-design.md",
    ];
    for path in in_scope_files {
        let contents = read(path).to_lowercase();
        assert!(
            !contents.contains("zerohunt"),
            "{path} still references the old project name \"zerohunt\""
        );
    }
}