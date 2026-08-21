# The benchmarking protocol

*A number from this repository means a min-of-N across interleaved builds on an
idle box, or it means nothing.*

The box these numbers come from: AMD Ryzen 7 8700F (8 cores, 16 threads),
Fedora 44, rustc 1.97.1 with LLVM 22.1.8.  A ratio from a different machine is a
different claim.

## Stop the poker solver first

The development box runs a poker solver that will happily take every core.  A
measurement taken beside it is worthless, and killing it unasked is worse.

```sh
pgrep -af poker
uptime          # the one-minute average should be near zero
```

If it is running, ask before touching it.  Proceed only once the box is idle.

## Interleave the builds, never run A then B

Build both sides first, stash the binaries, and only then measure — alternating
A, B, A, B for at least 15 passes each, all on one pinned core.  Cargo leaves
`.d` files and stale hashes beside the bench executable, so pick it explicitly
rather than globbing:

```sh
STASH=$(mktemp -d)
FILTER='(F8E4M3FNUZ|F8E5M2FNUZ|F8E4M3B11FNUZ|F8E4M3)/(add|sub|mul)/soft'  # regex over the bench id

pick() { find target/release/deps -maxdepth 1 -type f -executable -name "$1-*" \
           -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2; }

cargo build --release --benches && cp "$(pick arith)" "$STASH/arith-before"
# ... apply the change ...
cargo build --release --benches && cp "$(pick arith)" "$STASH/arith-after"

for i in $(seq 1 15); do
  CRITERION_HOME=$STASH/crit/A-$i taskset -c 2 "$STASH/arith-before" --bench "$FILTER"
  CRITERION_HOME=$STASH/crit/B-$i taskset -c 2 "$STASH/arith-after"  --bench "$FILTER"
done
```

A compile between two measurements heats the box, and a box that drifts over
fifteen minutes will hand you whichever answer the drift had at the time.
Interleaving cancels the drift instead of hoping it is not there.  `taskset`
pins both sides to the same core so neither can win by landing on a better one;
core 2 is an arbitrary choice, held constant.

Held constant *within a comparison*, not across sessions.  On 2026-08-21 two
agents &mdash; one on this crate, one on the C++ sibling &mdash; both read this
line, both ran `taskset -c 2`, and collided on the one physical core this
document names.  Check the core is free before pinning to it:

```sh
ps -eo psr,pcpu,args --sort=-pcpu | awk '$1==6'      # who is on core 6
cat /sys/devices/system/cpu/cpu6/topology/thread_siblings_list
```

Take the SMT sibling too, or the other thread of your core is someone else's.
And do not gate on `/proc/loadavg`: it counts runnable tasks absolutely, not per
core, so on this 16-thread box one pinned neighbour already reads 1.0 and a
threshold near zero can never clear.  The question is whether *your* core is
free, which loadavg does not answer.

## Take the minimum, not the mean

Noise on a benchmark is one-sided: nothing makes a loop run faster than it can,
and everything else on the machine makes it run slower.  The minimum across
passes is the least contaminated sample there is.  A mean is a statement about
the machine's other tenants.

Concretely: harvest `slope.point_estimate` from each pass's
`<CRITERION_HOME>/<group>/<bench>/new/estimates.json`, falling back to `mean`
where criterion reports no slope, and take the minimum **across passes** per
benchmark.  Criterion has no min-of-N mode of its own — its per-run figure is
already an aggregate, and this protocol treats each run as one sample.

That the minimum sheds a transient rather than averaging it in was measured on
2026-08-21, when a sustained all-core build from another session landed across
passes 7&ndash;9 of a fifteen-pass sweep.  Harvesting all fifteen and harvesting
with those three dropped gave 0.626x and 0.624x on the measured route, with the
control unmoved at 1.006x either way.  Before that this section argued
one-sided noise and cited nothing.

## Keep a control route

Every sweep carries at least one benchmark the change cannot possibly have
touched.  If the control moves, the run is noise and the headline number is
noise with it.  In this round's subtraction sweep the control was `mul` on the
same four shapes; it came back at 0.997–1.008x while `sub` moved to 0.773x,
which is what makes the 0.773x reportable.

## The noise floor is not a constant

Variants that changed nothing on the measured path have come back at 0.98x on
this box (6520d5a), and this section used to stop there: a ratio inside
`[0.98, 1.02]` is not a result.  That bounds *timing* noise.  It does not bound
code placement, which is larger, and which interleaving cannot touch.

Measured on 2026-08-21, on the exact-conversion round.  `benches/arith.rs`
numbers its closures `shape * 12 + op * 3 + {soft, f32-arm, f64-arm}`, the
un-instantiated route arm simply absent, so every bench body is individually
addressable in the symbol table.  Diffing the two binaries body by body, with
branch targets and rip-relative displacements normalised, gave 56 soft bodies
byte-identical and 56 hardware bodies changed — exactly the split the change
predicts.  `.text` shrank 2816 bytes and relocated all 112.  Those 56
byte-identical soft rows then measured **0.962x to 1.090x**, min of fifteen
interleaved passes.  Nine percent, on code that did not change an instruction.

No number of passes fixes this.  Placement is a property of the binary, not of
the run, so re-running measures it again rather than testing it — which is how
the C++ sibling came to report a cross-compiler disagreement whose whole
evidence was that it reproduced across two runs of the same two binaries
(retracted, 855686c).

So the control rows are not a sanity check.  They are a **calibration**: their
spread is the band this build pair can produce on rows of this duration, and a
per-row claim is reportable exactly when it clears that band.  This round's
hardware rows spanned 0.455x to 0.806x against a control band of 0.962x to
1.090x — disjoint, so every row stands on its own.  Where they overlap, only the
aggregate is reportable, and saying which one you have is the result.

The band is not reusable.  It varies with row duration *and* with how much of
`.text` the change moved: the C++ sibling measured 0.971x–1.025x and
0.886x–1.093x over the same 56 operator rows at the same durations, for two
different build pairs.  Those two halves are not equally solid, and it is worth
knowing which is which, because it is the pair of them that separates *varies by
build pair* from *varies by row duration*: 0.886x&ndash;1.093x rebuilds from two
commits, while 0.971x&ndash;1.025x was measured against a patch that was
reverted, so only one of its sides survives in the history (`../minifloat`
ca2d8cc, which says which rows are which and cites this file back).  Treat that
separation as indicated rather than settled &mdash; and if you measure a variant
you intend to throw away, commit it on a throwaway ref before you throw it away.

Calibrate every round regardless; a band inherited from another change is not
this change's band.  A null recorded is still worth more than a null dressed up.

## Corroborate from the other side

`cargo bench` is the second opinion, not the first.  It runs both targets:

- `benches/arith.rs` — each operator twice over the same operands, once as the
  crate computes it and once the way a caller would fake it.
- `benches/predicates.rs` — `is_nan`, `classify`, `partial_cmp`, `total_cmp`,
  which have no alternative route and exist to watch whether the bodies reach
  the caller at all.

Both run a deliberately short criterion window — 200 ms warm-up, 1 s
measurement, against criterion's 3 s and 5 s defaults.  Each iteration already
averages 1024 operations, so the estimate settles well inside that; the defaults
would turn 112 benchmarks into a quarter-hour run for no extra resolution.  It
is still the *second* opinion: a headline claim gets the interleaved treatment
above, and criterion is asked whether it agrees.

When the interleaved probe and the criterion geomean disagree in sign, neither
is reportable.  When they agree, quote both — 6520d5a quoted 1.130x against
1.126x geomean over 56 soft benchmarks.

For reference, the predicate side at e06641b (ns per element, one full pass):

| shape | `is_nan` | `classify` | `partial_cmp` | `total_cmp` |
| --- | --- | --- | --- | --- |
| `F8E4M3` | 0.423 | 0.496 | 1.131 | 0.659 |
| `F8E5M2FNUZ` | 0.214 | 0.445 | 0.808 | 0.660 |
| `F16` | 0.357 | 0.451 | 1.167 | 0.814 |
| `BF16` | 0.430 | 0.470 | 1.230 | 0.819 |

## The published page is a tripwire, not a measurement

Every push to `main` runs both bench targets on one `ubuntu-latest` runner at
`-Ctarget-cpu=x86-64-v3` and pushes the result to
<https://jdh8.github.io/minifloat-rs/dev/bench/>, one chart per benchmark id,
via `benchmark-action/github-action-benchmark`.  Nothing on that page is
hand-written; `.github/workflows/bench.yml` is the whole of it.

It is not a number by the standard at the top of this file.  A shared VM is not
an idle box, there is no interleaving, no control route, no pinned core, and one
sample per commit.  So a ratio read off that page does not go in a commit body
or in the changelog, and it cannot settle a claim against the interleaved
protocol above.  What it can do is notice a 2x cliff between two commits, which
neither of the other two routes does at all: the interleaved probe is the first
opinion, `cargo bench` on the named box is the second, and this is a third
thing that runs without being asked.  Hence `alert-threshold: 200%` with
`fail-on-alert: false` &mdash; the page shouts across a cliff and stays quiet
inside the runner's own spread.

The ISA is pinned rather than `native` because GitHub rotates runners across CPU
generations, and a series whose ISA changes under it is measuring the fleet.
That pins the code but not the silicon, and the runner's CPU is the one piece of
provenance the published data does not carry: `data.js` records commit, author,
message, and timestamp, and no CPU model or rustc version.  Trends across it are
readable; levels are not.

## An operator is timed only against a float that rounds like it

This is the rule behind `route` in `benches/arith.rs`, and it is a correctness
rule, not a benchmarking nicety: below 2*p* + 2 digits in the intermediate,
rounding twice can differ from rounding once.  [arithmetic.md](arithmetic.md)
states it in full, including the one place the implementation approximates it.
Do not restate it here; it has already drifted once from being written down in
four places.

## When the benchmark cannot see it

An inlining change is visible to `cargo bench` exactly when it publishes a body
rustc was not already carrying.  6520d5a marked the operators — big bodies,
which rustc's small-body heuristic declines — and `cargo bench` saw 1.126x.
1583802 marked the six `const fn` kernels, which rustc carries anyway at `-O2`
and above, and there was nothing to see: the downstream code is byte-identical.

For that second kind, reach for the disassembly instead: build a probe crate
with a path dependency and count the calls it emits, per
[inlining.md](inlining.md).  A count is exact, needs no idle box, and has no
noise floor.

Use the stopwatch for a change in what the code *does*, like the direct
subtraction, and the symbol table for a change in *where the code lives*.

## The non-arithmetic surface

*Standing decision, 2026-08-21: no route is built ahead of the round that needs
it.*

A claim about a classifier &mdash; `is_nan`, `is_infinite`, `is_finite`,
`classify`, the sign predicates &mdash; is settled by **counting, not timing**.
Each is a handful of bit operations, so at `-O3` there is nothing for a
stopwatch to find; what can go wrong is that the body stops reaching the caller,
and that is a count, per [inlining.md](inlining.md).  `benches/predicates.rs`
does time `is_nan` and `classify`, and the table above prints them &mdash; but
as the witness that both sides of an A/B build carry identical bench code, not
as the gate.

The comparisons *are* timed there for their own sake: `partial_cmp` and
`total_cmp` have rows on the same four shapes, and that is where a change to
them gets measured.  The converters have no route of their own: they ride the
hardware arm of
`benches/arith.rs`, whose every `f32` row is
`from_f32(x.to_f32() op y.to_f32())`, so a conversion regression does show up
there &mdash; mixed with the operator, which is the limitation and the reason a
standalone route will eventually be worth building.  It gets built by the round
that first optimizes a converter, not before.

When that round comes, the shape to copy is the Rust sibling,
[`jdh8/metallic-rs`](https://github.com/jdh8/metallic-rs): one bench file per
function over a shared harness, the competing implementations as sibling
criterion rows *in the same binary* &mdash; which is what `benches/arith.rs`
already does with its soft and hardware arms &mdash; and the input distribution
declared by the range form instead of left implicit.  Its publishing mechanism
has already been copied, minus its fan-out, by the section above, so the
reference is no longer only about file layout.  Its test side settles the
budget question that route will raise: a plain sequential traversal of all
2<sup>32</sup> `f32` bit patterns is routine practice there, so an exhaustive
sweep is affordable when an exhaustive sweep is what licenses the route.
`tests/all/ops.rs` now leans on the same fact &mdash; all 2<sup>32</sup> ordered
pairs of `F16` and `BF16` through `partial_cmp` and `total_cmp` cost 2.5 s of
the suite's 19 s, striped over the 32 threads of a Ryzen 9 7950X3D on
2026-08-21 &mdash; not the box in this file's header, which is why the machine
is named here.
