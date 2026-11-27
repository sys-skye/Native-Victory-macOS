# Coordinating several agents

When more than one agent (or human) works on this repository at the same time, follow this protocol so the work does not collide. Everything here is in addition to [AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md), which still apply.

## One branch per agent

- Work on your own fork, on a branch named `vibe/<short-slug>-<session-id>` (for example `vibe/flaky-probe-waits-63fece`), and open the PR from that branch against `main`.
- Never push to another agent's branch. Force-push only your own PR branch, and only with `--force-with-lease`.
- Keep your fork's `main` current with fast-forwards only (`git fetch <upstream> main && git push <fork> main`); never force it.

## Claim work before starting

- Comment `/claim` on the issue before starting anything big (CONTRIBUTING.md's rule; the `claimed` label and assignee come from `.github/workflows/claim.yml`), and open a draft PR with `Refs #<issue>` within 24 hours so two agents do not do the same work.
- One change per PR. Link the issue in the PR body; write `Closes #N` only when the PR fully resolves it.
- Pick from `docs/ROADMAP.md` or the `good first issue` / `help wanted` labels. If an issue is already claimed in the comments, leave it.

## Shared-machine etiquette (one Mac, several agents)

- Run one game at a time, serialised with a lock:
  `/usr/bin/lockf -k /tmp/victory-game.lock timeout 150 nice <run command>` (AGENTS.md).
- Keep builds polite: `-j6` or fewer, under `nice`.
- Logs, dumps, traces and image dumps go to a folder outside the repository; `cargo xtask policy` blocks anything game-derived from being committed.

## Hard rules (unchanged)

- Nothing from the game goes in the repo, and no game bytes to any external service or model.
- No AI attribution lines (`Co-Authored-By`, "Generated with ...") in commits or PRs.
- Guesses about undocumented console behaviour are marked in the code and in `docs/native-host.md`.
- New diagnostics are off by default, behind a `VICTORY_*` environment variable, and documented in `docs/native-host.md`.
- When the status changes, update `docs/native-host.md`, and `docs/ROADMAP.md` and the README if the roadmap moved.

## Before pushing a PR

Run what CI runs:

```bash
cargo test -p victory-hle -p victory -p victory-gpu -p xtask   # on the host
cargo clippy --workspace --all-targets --locked --target x86_64-pc-windows-gnullvm -- -D warnings
cargo xtask policy
```

Fill in the PR template's "How it was tested" and "Checklist" boxes — say which command you ran and how many iterations passed. The maintainer checks this before merging.

## Conflicts between agents

- Whoever merges second rebases their own branch onto the newer `main` and resolves the conflict; never rebase or rewrite someone else's branch for them.
- If two PRs touch the same lines, the smaller one lands first and the other rebases.
