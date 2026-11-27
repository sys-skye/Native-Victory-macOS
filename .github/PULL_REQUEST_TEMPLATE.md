## What this changes

<!-- One change per pull request. Name the issue you claimed for it: Refs #123 (or Closes #123). -->

## How it was tested

- [ ] `cargo test --target x86_64-apple-darwin -p victory-hle -p victory-loader -p victory`
- [ ] `cargo clippy --workspace --all-targets --locked --target x86_64-pc-windows-gnullvm -- -D warnings`
- [ ] Ran the game (say how far it got), or not needed

## Checklist

- [ ] No game files, game code, disassembly listings or SDK files are included
- [ ] Guesses about undocumented console behaviour are marked in the code and in docs/native-host.md
- [ ] New diagnostics are behind a `VICTORY_*` variable and documented
