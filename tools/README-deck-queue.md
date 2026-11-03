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

## toth needs the same thing, for a different reason

`tools/bench-lock` is the same ticket queue pointed at toth:

```bash
bench-lock run <name> -- <bench command>
```

The Deck's problem is that two runs measure each other. toth's problem is
worse: several agents build and bench here at once, and a timing taken at load
26 on 16 cores is measuring the other agents rather than the change under test.
So `bench-lock` admits one run at a time **and** waits for the 1-minute load
average to fall below `BENCH_LOCK_MAXLOAD` (default 4.0) before letting the
holder start.

It waits for a quiet gap; it cannot *keep* the machine quiet, so the queue is
still what stops someone starting a build on top of a measurement. Builds and
tests do not need it — only numbers you intend to quote.

The readiness check is the one generalisation this needed: `deck-queue` now
takes `DECK_QUEUE_RESOURCE` (a name for messages) and `DECK_QUEUE_READY_CMD` (a
command that must succeed before the holder runs). For the Deck that command is
an ssh login; for toth it is a load-average test.

## A quiet start is not a quiet batch

`tools/quiet-window` exists because `bench-lock` solves half the problem. The
lock waits for load to fall before letting a run start, but it cannot keep the
machine quiet for the five minutes a batch takes: another agent's build resumes
mid-batch, and the drift that causes is the same size as the effect under test.

Measured, not hypothetical: a real −3.1% on generation CPU had to be recorded
as a **null** because the base side drifted +3.2% monotonically across one
313 s batch. Three independent instruments agreed on 3–4.5% and the result
still was not quotable.

```bash
quiet-window open 30 <who>    # claim it, time-boxed
quiet-window check            # exit 0 == open; agents test this before building
quiet-window close
```

Two mechanisms, because neither alone is enough:

1. **A marker agents honour** before starting a build. Cooperative, and it is
   what keeps *new* work off the machine.
2. **`renice` to 19 on every compiler process already running.** Mechanical,
   needs nobody's cooperation, and lowering priority never needs privilege. A
   straggler mid-build keeps making progress and simply yields — nothing is
   paused and nothing is killed, so a window can never corrupt one agent's work
   to buy another a number.

Windows are time-boxed and expire on their own, so no agent can be held
indefinitely by a forgotten claim.

**What it does not do:** nice levels cannot isolate memory bandwidth or
last-level cache. A window is quieter, not sterile. Numbers taken in one should
say they were, and a residual drift inside a window is itself a finding — it
would mean prioritising is not enough and the answer is scheduling.
