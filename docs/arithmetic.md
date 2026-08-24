# The arithmetic

*Every operator works out a result exact enough to round correctly and rounds
it once.  All but the one shape that is `f32` do that on integer significands.
Nothing else in this document is as important as those sentences.*

## Integers, except for the host float itself

`+`, `-`, `*`, `/` used to evaluate in `f64` and round the answer back.  That is
only ever as good as `f64`'s reach, and a declared shape can outrun it: `E12M3`
squares 2<sup>&minus;1000</sup> to zero and 2<sup>1000</sup> to infinity through
an `f64`, though it represents both answers exactly (`test_arithmetic_past_f64`).

Apart from `E8M23`, the route is now `to_parts` → an exact integer computation
→ `from_parts`:

- **multiply** — two significands of at most 30 bits make a 60-bit product,
  which a `u64` holds exactly; exponents add.  Genuinely exact.
- **add** — `add_parts` aligns both addends on the lower exponent and sums them
  signed in an `i64`.  Addends more than `detail::ALIGN_CAP` = 32 binades apart
  drop the smaller one, which is under half the ULP of the larger and rounds
  straight back to it; that cap is also what keeps the aligned sum inside an
  `i64`.
- **divide** — `div_parts` normalises the dividend to bit 62, divides, and folds
  the remainder into the lowest bit as a sticky one.  Normalising is what
  guarantees the *p* + 2 quotient bits rounding needs; a fixed shift does not,
  and the next section says why.

So two of the four carry a deliberately inexact tail.  Neither tail can change
a rounding — that is what the cap and the normalisation are sized for — which is
why the thesis says *exact enough to round* and `add_parts` and `div_parts` say
the same in their own doc comments.

`from_parts` is the only place integer-route arithmetic rounds, ties to even,
by way of `detail::round_to_scale`.  One rounding, so no intermediate can lose
what the format is able to hold, and a shape whose exponent range overruns
`f64`'s is served as exactly as any other.  `E8M23` never reaches it: its FPU
operation is already at the target precision, and `from_fpu` only wraps the
result's bits.  The outbound conversion roundings remain fallback paths: the
`as f32` cast that ends `to_f32` where the shape does not fit, and `to_f64`
itself once a shape's exponent leaves `f64`'s range.  A shape the target holds
exactly takes neither.  `to_f32` and `to_f64` reassemble the target's bits and
round nothing at all.

Normal and subnormal destinations meet in one `from_parts` rounding path.  The
value exponent is clamped at the first normal row; any deficit extends the
rounding shift, while the row offset is zero below that boundary.  A carry then
crosses the subnormal/normal seam in the same contiguous code space as every
other carry.

In a dependent `-O3` baseline-x86-64 probe, the complete exported `F16`
multiplication body shrinks from 126 instructions at `67221dd` to 95 with the
merged path.  In the native arithmetic benchmark, `.text` falls from 3,375,956
to 3,368,356 bytes.  Across all 128 normalized iteration bodies, the instruction
count falls from 25,729 to 23,612 and conditional jumps from 2,375 to 2,218;
the old normal/subnormal paths merge around a clamp.  Some monomorphizations
lower it branchlessly, but there is no universal version of the C++ sibling's
`cmov` spelling.

Timing is mixed.  Min across fifteen interleaved passes pinned to CPU 5, the 64
soft rows have a 0.997x after/before geomean (0.935x&ndash;1.084x), while eight
unchanged matching-field controls span 0.994x&ndash;1.190x.  Of the 60 soft
bodies the merge changes, 32 clear that calibration band in the faster
direction and 28 overlap it; the four unchanged `E8M23` soft rows overlap too.
The aggregate is a recorded null, not a speed claim.  The eight standalone
conversion rows that reach `from_parts` have a 1.024x geomean
(0.950x&ndash;1.111x); their six controls span 0.996x&ndash;1.005x.  Ryzen 9
7950X3D, rustc 1.98.0 (LLVM 22.1.8), 2026-08-25, `-C target-cpu=native`.

The special cases stopped borrowing `f64`'s at the same time.  An invalid
operation yields the format's NaN, or `MAX` where the format has none, rather
than whatever sign the host's default NaN carried — `0 / 0` in a `Finite`
format used to be &minus;`MAX` on x86 and +`MAX` on ARM.

## The ceiling is 32 bits because the kernels are 64-bit

`Minifloat::Bits` is `u8`, `u16` or `u32`, and the last of those is where it
stops.  That is not a policy about how wide a minifloat ought to be: it is where
the `u64` kernels above run out.  With `E` ≥ 2 in 32 bits, `M` ≤ 29 and the
precision *p* = `M` + 1 ≤ 30, which every kernel has to survive.

| kernel | requirement | at *p* = 30 |
| --- | --- | --- |
| `mul`: exact product in a `u64` | 2*p* ≤ 64 | 60 bits |
| `add_parts`: a dropped far addend rounds back to the other | `ALIGN_CAP` ≥ *p* + 1 | 31 ≤ 32 |
| `add_parts`: the aligned sum fits an `i64` | *p* + `ALIGN_CAP` ≤ 63 | 62 bits |
| `div_parts`: *p* + 2 quotient bits to round with | dividend at bit 62 leaves ≥ 63 − *p* | 33 ≥ 32 |
| `round_to_scale`: a shift ≥ 64 yielding 0 is exact | every significand handed in ≤ 2<sup>63</sup> | product < 2<sup>60</sup>, quotient < 2<sup>63</sup> |

Any cap in [27, 37] passes every 32-bit shape, because no reachable shape makes
the addition bounds tight at once: a gap past the cap needs `E` ≥ 6, which
forces *p* ≤ 26.  `E2M29` has *p* = 30 but gaps of at most 2.  32 is what the
two bounds leave at *p* = 30 itself.

Division is the one that had to change rather than shrink.  A fixed quotient
shift of 63 − *p* is not enough: a subnormal dividend — a significand as small
as 1 — over a full-width divisor leaves only 63 − 2*p* quotient bits, below
*p* + 2 once *p* > 20.  Normalising the dividend leaves at least 63 − *p*
whatever it was.  Both facts are gated rather than argued: putting the fixed
shift back fails `test_arithmetic_correctly_rounded_wide` at `E8M22`, one ULP
out, and putting `ALIGN_CAP` back to 46 fails the same test where the aligned
sum overflows.  Neither mutation is visible to any 16-bit shape.

The correctness is not free.  On 2026-08-23, on an AMD Ryzen 9 7950X3D with
rustc 1.98.0 (LLVM 22.1.8), the min of fifteen interleaved passes comparing
`f51d531^` with `f51d531` put division at **1.064x** after/before across the
fourteen existing shapes, from 1.035x to 1.107x.  `mul`, which the change cannot
touch, spanned 0.995x to 1.007x.  Normalising the dividend therefore costs about
6%, outside the control band on every shape.

`BF20` exists only on the after side, so its second opinion is within the final
binary rather than an A/B build.  One `cargo bench -- BF20` pass over the same
operands gave:

| `BF20` operator | soft / `f64` |
| --- | --- |
| add | 1.272x |
| sub | 1.326x |
| mul | 0.814x |
| div | 0.899x |
| **geomean** | **1.054x** |

The integer route is still the faster bet for `BF20` multiplication and
division, not for addition, subtraction, or the four-operator geomean.

One thing the kernels do not bound is the exponent.  `integer_decode` returns an
`i16`, so `__minifloat!` asserts that both ends of the ULP scale fit one —
`MIN_EXP - MANTISSA_DIGITS ≥ i16::MIN` and `MAX_EXP - MANTISSA_DIGITS ≤
i16::MAX`.  The condition is on the range and not on `E`, because a custom bias
slides the range under any exponent width; with the default bias any `E` up to
15 passes.  Every `i32` exponent sum an operator forms is then trivially safe.

## Why there is no hardware route left to choose but one

There used to be `USE_F32_ADD` and `USE_F32_MUL` selecting between two routes
required to agree.  They are gone, and the survivor is not the slower one.
`benches/arith.rs` times each operator twice over the same operands — once as
the crate computes it, once the way a caller would fake it.

As of 2026-08-21, after the exact conversions below, the integer route wins
**42 of 56** comparisons, geomean **1.110x** in its favour, from 0.74x
(`E11M4` addition) to 1.44x (`E2M13` multiplication).  This paragraph used to
say 56 of 56 at 1.709x, and reading the fall as a regression would be exactly
backwards: no operator changed.  `to_f32` and `to_f64` did, and they belong to
the *comparator* &mdash; the soft arm of `bench_op!` is the bare operator and
calls neither.

The two halves were measured apart so the direction is not in doubt.  Min of
fifteen interleaved passes over the same binaries: the hardware route fell to
**0.626x** (0.455x to 0.806x) while the soft route held at **1.006x**, its 56
bench bodies confirmed byte-identical across the pair in the symbol table.  A
separate three-pass full-roster run reproduces both to three decimals on the
shapes they share, and its own before-side reproduces the old 1.709x as 1.703x.

All fourteen losses are additions or subtractions &mdash; nine and five &mdash;
the operators that do the most work against a round trip that does the least.
Nothing was traded for them.  They are the same operators at the same speed,
now measured against a comparator that no longer pays for an `exp2i` and an
int&rarr;float convert on every operand and every result.

The speed is a bonus.  The reason is correctness: the hardware route cannot
referee a shape it cannot hold, so keeping it would have meant keeping a route
that is wrong for exactly the shapes this crate exists to support.

### Matching exponent fields make conversion a shift

Default-biased IEEE E8 shapes have `f32`'s exponent and special-value fields;
default-biased IEEE E11 shapes have `f64`'s.  The crate's 32-bit ceiling leaves
E8M1 through E8M23 and E11M1 through E11M20.  Conversion out shifts the whole
code into the host layout.  Conversion in drops the unused mantissa bits with
an integer round-to-nearest-even bias, so no host arithmetic or floating-point
rounding mode participates.

This is only a conversion route; every narrower shape keeps its integer
arithmetic.  Input NaNs retain their sign and canonicalize their payload before
the rounding bias can erase it into infinity; conversion out preserves the
stored sign and payload because the entire code shifts unchanged.

### Nested exponent ranges make conversion a rebase

An exact shape whose `TINY / 2` is at or above the host's least normal value
takes a second integer route.  Its finite values nest inside the host's normal
range, so every host subnormal is below that midpoint and rounds to zero.  Every
remaining input is encoded by rebasing the host exponent field onto the
destination rows, extending the right shift when that row is
subnormal, and rounding the whole unsigned code.  Mantissa carry therefore
walks into the next exponent row without a special case.

The `f32` tier covers `F16`, every predefined 8-bit-and-below shape, and custom
shapes under the same range bounds, including equal-precision ones.  The `f64`
tier additionally covers `BF16`, `BF20`, and the wider exact shapes whose
exponent fields do not match `f64`'s.  Formats reaching past the host, and
shapes whose `TINY` reaches into its subnormal range, keep the generic
`decompose` → `from_parts` route.
Both fast tiers use only integer and bit operations, so DAZ, FTZ, and the host
rounding mode cannot reach them.

`test_nested_exponent_fast_path_exhaustive` walks all 2<sup>32</sup> `f32`
patterns through six representative formats, then checks an odd-stride sample
of the wider/custom-bias and zero-mantissa edges against the independent
reference encoder.  A separate raw-`u64` random sweep covers the `f64` tier;
explicit pairs on both sides of each host's subnormal gate pin its boundary.

The same dependent probe puts `F16` `from_f32` at 110 → 39 instructions and
`from_f64` at 98 → 43; `F8E4M3` moves 109 → 39 and 97 → 43.  `BF16` and `BF20`
keep their 15-instruction matching-field `from_f32`, while their `from_f64`
bodies move 98 → 43 and 100 → 44.  Ryzen 9 7950X3D, rustc 1.98.0 (LLVM
22.1.8), 2026-08-25, no LTO and `-C target-cpu=x86-64`.

The stopwatch agrees.  This is final / merged-rounding-only, min across fifteen
interleaved passes pinned to CPU 5, with sibling 21 free:

| encoding row | `BF16` | `BF20` | `F16` | `F8E4M3` | `F8E5M2FNUZ` | geomean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct `from_f32` | control | control | 0.654x | 0.689x | 0.766x | 0.702x |
| `from_f64` (`generic` ID) | 0.728x | 0.813x | 0.778x | 0.811x | 0.848x | 0.795x |

All eight affected rows clear the six unchanged controls' 0.998x&ndash;1.003x
band; together they have a 0.758x geomean.  Ryzen 9 7950X3D, rustc 1.98.0
(LLVM 22.1.8), 2026-08-25, `-C target-cpu=native`.

The earlier matching-field conversion round put the direct route against the
then-generic one in the same binary.  Min across fifteen alternating passes
pinned to CPU 5 of a Ryzen 9 7950X3D, with sibling 21 free, rustc 1.98.0 (LLVM
22.1.8), 2026-08-24:

| direct / generic | `BF16` | `BF20` |
| --- | ---: | ---: |
| `from_f32` | 0.213x | 0.217x |
| `to_f32` | 0.307x | 0.309x |

For `from_f32`, the `generic` row was the route that matching-field change
replaced.  Nested `from_f64` encoding now makes that historical benchmark ID a
misnomer on all five rows; it is retained so the series does not fork.  The
table predates that second route and is not a current direct / generic
comparison.  The `to_f32` comparator is an exact `f64` round trip, not the
previous exact reassembly.  A dependent `-O3` baseline-x86-64 probe supplied
that side without timing noise: `BF20` `from_f32` shrank from 101 instructions
to 15 and `to_f32` from 27 to 3; E11M20 `from_f64` shrank from 94 to 16 and
`to_f64` from 26 to 3.

### `E8M23` is `f32`, so the crate hands it to the FPU

`Minifloat::IS_F32` is stronger than exact conversion.  It requires the
precision, exponent range and non-finite semantics all to match `f32`; among
the shapes `minifloat!` admits, only S1E8M23 in the IEEE format with bias 127
does.  Its `to_f32` is therefore a bit-cast, its `from_f32` wraps the same bits
while canonicalizing a NaN payload, and its four operators are the corresponding
`f32` instructions followed by `from_fpu`.

The wrapper is load-bearing for NaNs.  An FPU may choose a NaN sign and payload,
and an invalid operation may choose a different default on x86 and ARM, while
the crate promises its positive `NAN` for arithmetic.  `from_fpu` replaces any
hardware NaN with that one encoding; ordinary values, signed zero, infinities
and subnormals pass through bit for bit.  `from_f32` separately preserves the
input NaN's sign while canonicalizing its payload, as every conversion into a
minifloat does.

Unlike the exact conversion paths above, the arithmetic operation here really
does reach the FPU.  On targets that lower it to hardware `f32` instructions,
it follows the target's rounding mode and DAZ/FTZ settings.  Rust's float
semantics assume the default FP environment, so this is not an additional
portable guarantee; the distinction reaches only a caller already outside
`f32`'s contract through FFI.

Measured from benchmark-only parent `79dcb9e` to this direct-route commit on
2026-08-23 on an AMD Ryzen 9 7950X3D with rustc 1.98.0 (LLVM 22.1.8) and
`-C target-cpu=native`, min of fifteen interleaved passes pinned to CPU 6 with
its SMT sibling free, the direct route costs this fraction of the old integer
route:

| `E8M23` operator | after / before |
| --- | ---: |
| add | 0.087x |
| sub | 0.098x |
| mul | 0.144x |
| div | 0.365x |
| **geomean** | **0.146x** |

One separate Criterion pass of the `arith` target corroborated the direction
at 0.087x, 0.099x, 0.143x and 0.366x respectively, also 0.146x at geomean.

The 28 unchanged rows &mdash; both arms of `F16`, `BF16` and `BF20`, plus
`E8M23`'s `f64` arm &mdash; span 0.981x&ndash;1.029x.  A normalized symbol audit
agrees with the experiment: 124 of the 128 iteration bodies are identical, and
the only four that changed are these `E8M23` soft rows.  Every result therefore
clears this build pair's control band, division by a factor of 2.7 at the
closest.

Identity does not license routing every narrower shape through a host float.
Such a route is possible where an exhaustive proof covers it, but adopting one
per operator and per shape would split an engine that currently has one path
and spend that shape's independent regression detector.  `BF16` already shows
the proof burden: its subnormals leave `f32` only 16 digits where the 2<var>p</var>
+ 2 rule asks for 18, and only the exhaustive sweep licenses the benchmark's
pairing.  A measured round can make that trade; this identity does not make it
for free.

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
The wide roster — nine 16-bit shapes and six on `u32`, `BF20` among them — is
sampled at 2<sup>16</sup> pairs, plus 2<sup>12</sup> whose left operand is
subnormal, which is the case `div_parts` normalises for and which a uniform draw
over a 32-bit shape practically never hits.  2<sup>13</sup> uniform pairs turned
out to miss the double rounding an `f32` intermediate causes in `E2M13`, which
is why the count is what it is.  Those shapes are sampled by *code* as well:
above 20 bits `support::codes` walks an odd stride landing near a million, odd
so every low-bit pattern and hence every tie parity is still visited.

`E8M23` now has two referees with different jobs.  The wide exact-oracle sample
independently checks its rounding.  `test_f32_shape_matches_hardware` runs
2<sup>22</sup> pairs through the direct route and checks its plumbing against
the FPU's own bits — signed zero, infinities and subnormals pass straight
through, while every NaN must return the crate's one canonical encoding.  Four
deterministic invalid operations and a negative payload-carrying NaN pin the
case random operands practically never reach.  `test_f32_shape_matches_core`
pins the shape's constants against `core::f32`.

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

*Superseded 2026-08-21 &mdash; by a different question, which is why the lesson
stands.*  That null asked whether a *second* scale multiply could be skipped,
and its answer, that LLVM had already folded it, was correct and still is.
Reassembling the target's bits is not a larger version of that idea: it deletes
the int&rarr;float convert and the surviving multiply together, leaving a shift,
an add and a bit-cast, so there is nothing for LLVM to fold because there is no
arithmetic left.  Measured at 0.626x on the hardware route (above).  A null
about removing one instruction bounds nothing about removing the computation.

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

**LLVM leaves the two `add_parts` alignments scalar through x86-64-v3.**  On
2026-08-23, a dependent `-O3` probe of `fe5bd18` built with rustc
1.98.0 (LLVM 22.1.8) on the same AMD Ryzen 9 7950X3D contains no selected
packed-integer instruction with explicit `target-cpu=x86-64` or `x86-64-v3`.
Instruction counts for the complete exported addition body are:

| shape | x86-64 | x86-64-v3 | native AVX-512 |
| --- | ---: | ---: | ---: |
| `F8E5M2` | 130 scalar | 125 scalar | 116 vectorized |
| `F16` | 131 scalar | 126 scalar | 116 vectorized |
| `BF16` | 162 scalar | 149 scalar | 136 vectorized |

The native build shows that LLVM can pair the work when AVX-512 is available;
the crate's baseline and x86-64-v3 targets do not.  No code follows from a
disassembly result: whether the wider instructions win is a timing question
for a later round.

**Branchless selects for every exact conversion were not ported.**  The C++
sibling tried that spelling in `9bfa044`; its Clang build regressed to 1.513x
despite GCC improving to 0.897x.  Rust uses LLVM too, and Clang's unconditional
subnormal renormalization is the route that regressed.

**Lossy outbound integer-field rounding was not ported.**  The C++ sibling's
`195090c` wins were confined to E12M3-class outputs.  Those shapes are
test-local here, while the E11-to-`f32` shape Rust does share corresponds to
the route that regressed 2.2&ndash;3.3x there.  The existing exact and
matching-field outbound paths are already shifts; keep the generic fallback
until a dependent profile contains the one class it helps.

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
