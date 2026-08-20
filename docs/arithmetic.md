# The arithmetic

*Every operator works out a result exact enough to round correctly, on integer
significands, and rounds it once.  Nothing else in this document is as
important as that sentence.*

## Integers, not a hardware float

`+`, `-`, `*`, `/` used to evaluate in `f64` and round the answer back.  That is
only ever as good as `f64`'s reach, and a declared shape can outrun it: `E12M3`
squares 2<sup>&minus;1000</sup> to zero and 2<sup>1000</sup> to infinity through
an `f64`, though it represents both answers exactly (`test_arithmetic_past_f64`).

The route now is `to_parts` → an exact integer computation → `from_parts`:

- **multiply** — two significands of at most 15 bits multiply exactly in a
  `u64`, exponents add.  Genuinely exact.
- **add** — `add_parts` aligns both addends on the lower exponent and sums them
  signed in an `i64`.  Addends more than `detail::ALIGN_CAP` = 46 binades apart
  drop the smaller one, which is under half the ULP of the larger and rounds
  straight back to it; that cap is also what keeps the aligned sum inside an
  `i64`.
- **divide** — `div_parts` computes `detail::QUOTIENT_BITS` = 46 quotient bits
  and folds the remainder into the lowest as a sticky bit.  The two 46s are
  unrelated: one is an alignment window, the other a quotient width.

So two of the four carry a deliberately inexact tail.  Neither tail can change
a rounding — that is what the two constants are sized for — which is why the
thesis says *exact enough to round* and `add_parts` and `div_parts` say the same
in their own doc comments.

`from_parts` is the only place *arithmetic* rounds, ties to even, by way of
`detail::round_to_scale`.  One rounding, so no intermediate can lose what the
format is able to hold, and a shape whose exponent range overruns `f64`'s is
served as exactly as any other.  The crate's other two roundings are outbound
and unrelated: the `as f32` cast at the end of `to_f32`, and `to_f64` itself
once a shape's exponent leaves `f64`'s range.

The special cases stopped borrowing `f64`'s at the same time.  An invalid
operation yields the format's NaN, or `MAX` where the format has none, rather
than whatever sign the host's default NaN carried — `0 / 0` in a `Finite`
format used to be &minus;`MAX` on x86 and +`MAX` on ARM.

## Why there is no hardware route left to choose

There used to be `USE_F32_ADD` and `USE_F32_MUL` selecting between two routes
required to agree.  They are gone, and the survivor is not the slower one.
`benches/arith.rs` times each operator twice over the same operands — once as
the crate computes it, once the way a caller would fake it.

As of 2026-08-21, after the direct subtraction below, the integer route wins
**56 of 56** comparisons, geomean **1.709x** in its favour, from 1.01x
(`BF16` addition, the narrowest margin) to 2.63x (`F8E4M3FN` multiplication).
That is up from **53 of 56** at c4c28b1, whose three losses were all `FNUZ`
subtraction.

The speed is a bonus.  The reason is correctness: the hardware route cannot
referee a shape it cannot hold, so keeping it would have meant keeping a route
that is wrong for exactly the shapes this crate exists to support.

## The 2*p* + 2 rule, and why the benchmark skips shapes

A hardware float may stand in for a shape only if both operands are exact in it
**and** it carries at least 2*p* + 2 digits, where *p* is the shape's own
precision.  Below that, rounding to the intermediate and then to the shape can
differ from rounding to the shape once (Figueroa 1995).  This is the whole of
`route` in `benches/arith.rs`.

Exactness alone is not enough.  `E2M13` is exact in `f32` — 14 digits into 24 —
yet a product of two of its significands is 28 digits wide, so it goes through
`f64`.  `E11M4` is not exact in `f32` at all and falls back the same way.
`E12M3` reaches past `f64` altogether and is skipped rather than timed against a
different answer.

One caveat the code does not state: `route` compares `T::MANTISSA_DIGITS`, which
is the *normal-range* precision.  A subnormal has fewer digits, and `BF16` is the
shape where that bites — its subnormals leave `f32` 16 digits where the rule
wants 18.  What licenses that pairing is therefore not the rule but the
exhaustive check below, which found no disagreement.  `BF16` is the shape that
needed it.

That the `f32` route really does round alike is a gate of its own, in two tests:
`test_arithmetic_matches_f32`, over every ordered pair of every shape in the
8-bit test roster, and `test_arithmetic_matches_f32_16bit`, over all
2<sup>32</sup> ordered pairs of `F16` and `BF16`.

## The oracle shares no arithmetic with the implementation

The old correctness chain had two links — `from_f64` rounds correctly, and every
operator matches `from_f64(xf op yf)` — and nothing tested the join.  It also
could not outlive the `f64` route it was written against.

The oracle in `tests/all/arith.rs` — `check_pair` with `exact_sum`, `cmp_exact`
and `reference_round` — computes each result exactly and converts nothing to a
float: a product of significands, a signed sum aligned in an `i128`, a quotient
compared by cross-multiplication.  `reference_round` reaches its value only
through a comparison callback, so a value no float type can hold referees
itself.

Its coverage, stated precisely, because the loose version keeps getting
repeated: every **ordered pair of finite operands** of every shape in a
hand-written roster (`tests/all/support.rs`) of 34 declared types at widths 4, 6
and 8, spanning all four formats and exponent widths 2 through 6.  Pairs
involving an infinity or a NaN have no exact magnitude to compare against and
return early; those are pinned by `test_arithmetic_matches_f64` instead.
16-bit shapes are sampled at 2<sup>16</sup> pairs — 2<sup>13</sup> turned out to
miss the double rounding an `f32` intermediate causes in `E2M13`.

The older `f64` and `f32` comparisons stay in the file on purpose.  They are
narrower — they cannot referee a shape their float cannot hold — but they cover
the non-finite pairs the exact oracle skips, and they are the only check that the
hardware route the benchmark times computes the same answer.

## No lookup tables

Tempting for the 8-bit shapes: 2<sup>16</sup> entries per operator, one load
instead of a rounding.  Two reasons it is not there.

It does not generalize.  A `u16` shape needs 2<sup>32</sup> entries per
operator — 8 GB for `F16` addition alone — so a table would serve the ten
`u8`-backed predefined types and leave `F16` and `BF16` on the integer path,
which would have to exist anyway.  Any 16-bit shape a user declares lands there
too.

And the measurement would lie.  A microbenchmark hammers the same 64 KB in a
loop with nothing else in L2, which is the one condition under which a table
looks good; a caller doing anything else evicts it and pays a miss where the
integer route pays a shift.  A benchmark that cannot represent the failure mode
cannot be used to argue for the thing that fails that way.

## `FNUZ` subtraction, twice

**First finding (8d3bac7).**  `Sub` was `self + -rhs`, and a format without a
negative zero needs a guard in `neg`: the code a zero would flip into is its
NaN.  The `setcc` answering that question wrote a byte register, and a partial
write merges with what the register already held.  In a loop of subtractions
that register is where the last `add` left its sum, so the loop ran serialized —
`F8E5M2FNUZ` subtraction cost 17.0 ns against 11.2 ns for its own addition, one
of the three `FNUZ` subtractions a round trip through `f32` still beat.
Rewriting the guard as the top bit of `m | -m` opened it with a full-width `mov`
and no false dependency: 11.7 ns against 11.1 ns, at an unchanged instruction
count.

**Second finding (this round).**  A gap survived that fix — smaller, and
structural rather than microarchitectural.  `-rhs` still had to *produce a
representable value*, so `FNUZ` still paid for the guard, just without the
stall, and its subtraction still cost about 18% more than its own addition.
There is no reason for an intermediate to exist at all: subtraction is addition
with one sign inverted, and by the time `add_parts` sees a sign it is a `bool`.

`add_impl(self, rhs, flip)` inverts it there, with `Add` passing `false` and
`Sub` passing `true`.  Both call sites pass a literal, so the flag folds away
before anything is emitted, and no format special-cases anything.

Measured on an idle box under [benchmarking.md](benchmarking.md), min across 30
interleaved passes of alternating builds, `benches/arith.rs`, ns per element:

| | before | after | |
| --- | --- | --- | --- |
| `F8E5M2FNUZ` sub | 5.064 | 3.917 | 0.773x |
| `F8E4M3FNUZ` sub | 4.996 | 3.885 | 0.778x |
| `F8E4M3B11FNUZ` sub | 4.991 | 3.896 | 0.781x |
| `F8E4M3` sub | 3.664 | 3.372 | 0.920x |
| add, all four shapes | 4.22–4.30 | 3.95–4.02 | 0.93–0.94x |
| **mul, the control** | 2.90–3.18 | 2.90–3.18 | 0.997–1.008x |

`FNUZ` subtraction against its own addition goes from 1.18x to 0.98x.  Addition
got 7% faster as a side effect: the old body asked `self.is_infinite()` twice,
once in `a || b` and once to pick the answer, and `add_impl` asks each operand
once.

Sign-of-zero is the part worth checking rather than trusting.  With the flag
folded into `rhs_negative`, cancellation still yields +0 unless *both* addends
were negative: `0 − 0 = +0`, `(−0) − (−0) = +0`, `(−0) − (+0) = −0`, `x − 0 = x`.
That was verified exhaustively rather than argued — see below.

**What the fix left behind.**  `Sub` no longer reaches `Neg` at all; the only
surviving `-rhs` is the arm where the right operand is an infinity, which is
reachable only for `IEEE`, where the guard is not taken.  The `m | -m` trick in
`Neg` therefore no longer has the caller that made its stall visible.  It is
kept because it is still the cheaper way to ask the question for a user writing
`-x` on a `FNUZ` value, and its comment now says so rather than describing a
loop that no longer exists.

## Nulls from this round, recorded on purpose

**Skipping the second `exp2i` in `to_f64`: not needed.**  The plan was to branch
on `HAS_EXACT_F64_CONVERSION` and use a single scale factor.  The disassembly of
a dependent crate already shows one `vmulsd` for the scale: where the exponent
provably stays inside `f64`'s range, LLVM folds `exp2i(exponent − head)` to 1.0
and drops the multiply on its own.  The branch would have been dead weight.

**Fusing the `is_finite` comparison: not needed.**  `!is_nan() && !is_infinite()`
compiles to four instructions for `F16` —

```
not %edi ; test $0x7c00,%edi ; setne %al ; ret
```

— LLVM having recognized both halves as one question about the exponent field.
Nothing hand-written would improve on that.

**`#[inline]` on the kernels changes nothing at `-O2` and above.**  Byte-identical
downstream code.  It was kept for the `-O1` case and for not depending on a
compiler heuristic; the full argument is in [inlining.md](inlining.md).

## How the subtraction change was checked

Three gates, in order.

1. `cargo test --release` — the exact integer oracle, unchanged.
2. An equivalence sweep against the implementation being replaced.  Both
   spellings are public API, so `x - y` was compared against `x + (-y)` bit for
   bit from outside the crate: every ordered pair of 26 shapes at 8 bits
   covering all four formats and five exponent widths, and all 2<sup>32</sup>
   ordered pairs of eight 16-bit shapes — `F16`, `BF16`, a custom bias of 1000,
   and shapes reaching past `f64`.  Every pair agreed, in 25 seconds.
3. The interleaved benchmark protocol in [benchmarking.md](benchmarking.md).

The second gate is the one that made the change safe to make.  It proves the
refactor is behaviour-preserving rather than merely correct-looking.  It is a
one-off rather than a repository fixture, because it can only exist while both
spellings do; the standing gate is the oracle.
