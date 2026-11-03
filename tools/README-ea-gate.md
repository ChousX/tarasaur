# ea as a gate runner, with no credential on it

`tools/ea-gate` runs correctness gates — `cargo test`, `clippy`, `check` — on
**ea**, a 4-thread 8 GB always-on laptop that otherwise sits at load 0.3. The
point is that a quiet window on toth stalls five agents' compiler work for half
an hour; ea absorbs it instead.

```bash
ea-gate setup [repos...]                 # clone/refresh + install the runner
ea-gate run <repo> <ref> -- <cmd...>     # one at a time, nice 19, -j 3
ea-gate status | log | preflight
```

ea is **not** a measurement host and `run` refuses commands that look like
benchmarks. It is a different CPU from toth and it runs Kodi and Jellyfin, so
any timing from it is incomparable to `bench/LOG.md` and to the Deck.

## The interesting part: getting a toolchain without giving ea credentials

These projects' devenv shells import `chousx/devenv/bevy` from
`git+ssh://git@github.com/ChousX/NixOS`, which is **private** — the public API
404s — and ea has no github credentials, so the shell could not build and there
was no cargo. Three things were tried:

1. **`nix copy` the locked input into ea's store.** Not enough: nix's git
   fetcher still contacts github to resolve `ref=refs/heads/main` even with the
   rev pinned.
2. **Agent forwarding, or a deploy key on ea.** Would work, and was authorised,
   but it puts a credential (or a live agent socket) on a box that boots into
   Kodi and serves Jellyfin on the LAN.
3. **What was actually done: serve the input from toth.** A `--mirror` clone of
   toth's own `~/.nixos` lives on ea, and one `insteadOf` rewrite points
   `ssh://git@github.com/ChousX/NixOS` at `file://` that mirror. nix then
   resolves the input locally and never reaches github. **No credential, not
   even in flight.**

Two details that each cost an attempt:

- **`file://`, not a plain path.** nix asks for `shallow=1`, and git's plain-path
  "local" transport cannot serve a shallow fetch; the `file://` smart transport
  can.
- **The rewrite has to be visible to `direnv`, not to the command.**
  `direnv exec . env GIT_CONFIG_GLOBAL=… cmd` sets it for `cmd`, but nix fetches
  while *direnv* builds the shell — too late. It goes before `direnv`.

ea's own global git config is on a read-only declarative filesystem, which is
why the rewrite lives in a separate file reached only through
`GIT_CONFIG_GLOBAL` inside a gate run. Nothing outside `ea-gate` sees it.

Everything placed on ea is listed, with its removal command, in the session's
`EA-CLEANUP.md`.

## Why the work is done by a runner installed on ea

Quoting a command through `bash -> ssh -> bash -> direnv -> bash` broke twice:
once mangling the command outright, and once letting `git checkout --detach … |
tail -2` swallow a failed checkout's exit status, so the gate ran against the
clone's default HEAD and **reported success**. The command now travels
base64-encoded to a runner on ea which resolves the ref itself, never pipes the
checkout, and runs under `bash -eo pipefail -c` — because a gate that returns 0
while failing is the exact hazard this tooling exists to catch.
