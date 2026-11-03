# One benchmark at a time on the Deck

`tools/deck-queue` and `tools/deck-ssh` make the Steam Deck's single-occupancy
rule mechanical instead of remembered. Install them on `PATH`
(`~/.local/bin` is where they run from on toth) and every agent that wants the
Deck takes a ticket:

```bash
deck-queue run <name> -- tools/bench-deck-ab.sh <args...>
```

called as a **background** command, because it blocks while it waits its turn.

## Why it exists

Two rules were being carried by discipline alone, and both had already been
broken by accident:

- **Two runs at once measure each other.** Nothing stopped a second agent
  starting a run while one was in flight.
- **An ssh login during a run shows up in the measurement.** It lands in the
  `other` CPU column the disturbance verdict reads, and rejected run `ab2-A1`
  at 18.4% foreign CPU during `spin`. The driver now reports back over a TCP
  callback so there is no need to look — but nothing stopped anyone looking.

So `deck-ssh` **refuses** unless the caller holds the queue's claim
(`DECK_QUEUE_TICKET` is exported only inside a ticket), and the queue admits
one holder at a time.

## What it guarantees

- **Strict FIFO by ticket**, verified: four concurrent submissions ran
  1→2→3→4 with no overlap in their start/end stamps.
- **No wedging.** Every waiter prunes tickets and claims whose process is
  gone, so a killed agent releases the Deck rather than holding it forever.
- **A real login check, not a port check.** Tailscale SSH can hold port 22
  open and still refuse until the human clears a browser check, so the
  preflight runs `ssh ... true` rather than opening a socket.
- **It holds its place while the Deck is away.** If the Deck is unreachable the
  holder keeps its claim and waits (`--deck-timeout`, default 1h), because
  nobody else can use the Deck either and losing the place would reorder the
  night's measurements for nothing. It exits 75 (`EX_TEMPFAIL`) if it gives up:
  the work was fine, the instrument was not.

## Reading it

```bash
deck-queue status          # who holds the Deck, who is waiting
deck-queue status --json
deck-queue log 40          # what has run, with exit codes and durations
deck-queue preflight       # is the Deck reachable and will it let us in
deck-queue cancel <ticket>
```

Per-run logs land in `~/.local/state/deck-queue/runs/`.

## What it does not do

It does not pick a result name for you. `bench_results/` is shared across
machines and the findings tables glob by name — there are already two
different run sets both called `pool-A1` — so that is still yours to get
right.
