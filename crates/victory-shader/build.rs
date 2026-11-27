//! `VICTORY_SHADER_SOURCE_HASH`: what the translator is built from (this
//! crate's sources but its `*_tests.rs` test modules, manifest and build
//! script, the lock file's entries of the packages it depends on and the
//! compiler's version), for caches of translations
//! (`victory_shader::SOURCE_HASH`). The rest of the lock file is left out: a
//! dependency of another crate does not change a translation, and hashing the
//! whole file made every such change cost every cached translation (#334).

use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};

include!("lock.rs");

fn main() {
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let mut files = vec![dir.join("Cargo.toml"), dir.join("build.rs"), dir.join("lock.rs")];
    sources(&dir.join("src"), &mut files);
    files.sort();
    // SipHash with fixed keys: stable for one compiler, whose version is
    // hashed too.
    let mut h = DefaultHasher::new();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output();
    h.write(&version.map(|o| o.stdout).unwrap_or_default());
    for f in &files {
        let name = f
            .strip_prefix(&dir)
            .unwrap_or(f)
            .to_string_lossy()
            .replace('\\', "/");
        h.write(name.as_bytes());
        h.write(&std::fs::read(f).unwrap_or_default());
    }
    let lock = std::fs::read_to_string(dir.join("../../Cargo.lock")).unwrap_or_default();
    for entry in dependency_entries(&lock, "victory-shader") {
        h.write(entry.as_bytes());
    }
    for watched in ["src", "Cargo.toml", "build.rs", "lock.rs", "../../Cargo.lock"] {
        println!("cargo:rerun-if-changed={watched}");
    }
    println!(
        "cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={:016x}",
        h.finish()
    );
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else if !path.to_string_lossy().ends_with("_tests.rs") {
            // Test modules (`*_tests.rs`, declared under `cfg(test)`, which
            // `tests::test_modules_are_test_only` checks) do not change the translator.
            out.push(path);
        }
    }
}
