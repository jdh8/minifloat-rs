# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] &mdash; 0.3.0

### Added

- The `Finite` format: every bit pattern is a number, with neither infinities
  nor NaN.  This is what the OCP MX sub-8-bit types actually use.
- LLVM's set of floating-point types no wider than 16 bits is now real public
  API in the crate root: `F4E2M1FN`, `F6E2M3FN`, `F6E3M2FN`, `F8E3M4`,
  `F8E4M3`, `F8E4M3FN`, `F8E4M3FNUZ`, `F8E4M3B11FNUZ`, `F8E5M2`, `F8E5M2FNUZ`,
  alongside the existing `F16` and `BF16`.
- Inherent `HAS_INF`, `HAS_NAN`, and `HAS_NEG_ZERO` constants on every generated
  type, plus an inherent `MAX_10_EXP`.
- `Minifloat::INFINITY` and `Minifloat::HAS_NEG_ZERO`.  Generic code could not
  reach an infinity at all before.
- The public `Format` enum, an inherent `FORMAT` constant on every generated
  type, and `Minifloat::FORMAT` for generic code.  `Format::has_inf`,
  `Format::has_nan`, and `Format::has_neg_zero` answer what a format is without
  naming a type.
- A `cargo bench` target timing each operator against the route it replaced:
  widen both operands to a hardware float, let the FPU work, round the result
  back.  A shape is timed only against a float that rounds the way it does
  &mdash; `f32` where 2<var>p</var> + 2 of its digits fit, `f64` otherwise, and
  no comparison at all for a shape reaching past `f64`.  That the `f32` route
  really does round alike is now checked in its own right, exhaustively over
  every pair of every shape in the 8-bit test roster and over all
  2<sup>32</sup> pairs of `F16` and `BF16`.
- A second `cargo bench` target, `predicates`, timing `is_nan`, `classify`,
  `partial_cmp` and `total_cmp`.  These have no alternative route to be
  compared against; the target exists so an inlining sweep can be measured with
  identical bench code on both sides.
- `docs/`, three standing references that outlive the commits they came from:
  `arithmetic.md` on why every operator rounds once on integer significands and
  what the oracle actually covers, `inlining.md` on what a dependent crate gets
  to inline and what rustc already does for free, and `benchmarking.md` on the
  protocol behind every number in this project.  `README.md` links to them.
- `CLAUDE.md`, a routing table into those documents plus the standing rules.
- A published benchmark page, <https://jdh8.github.io/minifloat-rs/dev/bench/>,
  refreshed on every push to `main` by `.github/workflows/bench.yml`.  It is a
  trend line rather than a claim: one shared runner, no interleaving, no control
  route, one sample per commit, so a ratio read off it does not go in a commit
  body or in this file.  It exists to notice a 2x cliff between two commits, and
  `docs/benchmarking.md` says so where it says what a number here means.
- The comparison layer is now gated the way the arithmetic already was.
  `total_cmp` had no test at all; it now faces every ordered pair of all 34
  shapes in the 8-bit roster, refereed by properties rather than by a second
  copy of the key: the numeric half of the expectation comes from
  `partial_cmp`, and only what a value comparison cannot express &mdash; the
  ±0 split, and where each format parks its NaN &mdash; is stated in encoding
  terms.  `partial_cmp` against an `f32` referee grows from that roster to all
  2<sup>32</sup> ordered pairs of `F16` and `BF16`, 2.5 s of the suite's 19 s.
  `const_eq` and `const_partial_cmp` are swept against `==` and `partial_cmp`
  over every pair of one shape per format, where they had been `F16` spot
  checks; `abs` and the sign predicates get their first assertions; and
  `is_finite` joins the one-hot `classify` sweep.
- `Minifloat::total_cmp` now documents that the order is on the *encoding*, so
  an `FNUZ` NaN &mdash; which occupies the &minus;0 slot &mdash; orders between
  the negative numbers and +0 rather than beyond ±`MAX`.  Behaviour is
  unchanged; the new sweep pins it.

### Changed

- **The default exponent bias of `FNUZ` is now 2<sup>E&minus;1</sup>** instead
  of 2<sup>E&minus;1</sup> &minus; 1, matching LLVM.  Invocations that spell the
  bias out are unaffected.
- `Minifloat::NAN` is now an `Option<Self>`, symmetric with the new
  `Minifloat::INFINITY`.  The inherent `NAN` is unchanged where it exists, and
  is no longer declared for formats without a NaN encoding.
- `Minifloat::MAX_EXP` and `Minifloat::MAX_10_EXP` are now required constants
  with no default; the `minifloat!` macro fills them in.
- The types formerly in the `example` module are promoted to the crate root as
  supported API.
- The `minifloat!` macro is now a thin format layer over a `__minifloat!`
  engine.  Each format arm now passes one `Format` variant instead of five
  literals; the engine derives `HAS_INF`, `HAS_NAN`, `HAS_NEG_ZERO`, `HUGE`,
  and the NaN bit pattern from it.
- `Minifloat` gains a required `FORMAT` constant, and its `HAS_NEG_ZERO`
  constant now defaults from `FORMAT`.
- `Minifloat::Bits` is now sealed to `u8` and `u16`.  A minifloat is a sign bit
  plus fewer than 16 magnitude bits, so no wider storage was ever reachable.
- **Binary `+`, `-`, `*`, `/` no longer evaluate in a hardware float.**  Each
  operator works out the result on integer significands, exactly enough to round
  it once, so no intermediate can lose what the format is able to hold.  An
  invalid operation (0/0, ∞ &minus; ∞, ∞ &times; 0, ∞/∞) yields the format's
  NaN, or `MAX` where the format has none, instead of inheriting the host's
  default NaN sign.  Correct rounding is now a promise for *every* shape the
  macro accepts, including one whose exponent range overruns `f64`'s, which the
  old route flushed to zero or to infinity on the way through.
- Negation no longer decides with a comparison whether to flip the sign.  Only
  a format without a negative zero has to ask &mdash; the code a zero would flip
  into is its NaN &mdash; and the `setcc` answering it wrote a byte register,
  inheriting a false dependency on whatever was last in it.  That cost `FNUZ`
  subtraction 17 ns against 11 ns for its own addition, back when subtraction
  was a negation followed by an addition; asking in arithmetic instead brought
  it to 11.7 ns.  Subtraction no longer goes through negation at all (below),
  so what remains here serves a caller writing `-x`.
- **Every concrete method in the crate is now `#[inline]`** &mdash; the four
  operators and their compound forms, `Neg`, the rounding core, both conversions
  in each direction, the six kernels in `detail`, every predicate and comparison
  the `minifloat!` macro generates, the `PartialEq` / `PartialOrd` / `Hash`
  impls, the `Minifloat` forwarders, and `Format`'s own three predicates.
  Nothing here is generic, so a crate that depends on this one used to reach
  each of them through a call, and a dependency cannot ask for LTO on its
  consumer's behalf: Cargo reads a profile only from the workspace root and
  silently ignores one in a dependency.

  Marking part of the set is worse than marking none of it &mdash; the caller
  inlines `add`, whose body still calls out to `from_parts` and `to_parts`, so
  one call becomes two &mdash; which measured 0.99x against the 1.13x of the
  completed operator set, and 1.10x with the conversions along.  Measured again
  this round by counting instead of timing, on a probe crate with a path
  dependency and 34 entry points: call instructions emitted downstream fall from
  62 to 0 at `opt-level = 1` and from 4 to 0 at `opt-level = 3`, where marking
  the kernels alone leaves 46 and the predicates alone 16.  From `-O2` up most
  of this is redundant with rustc's own small-body heuristic and the MIR it
  publishes for a `const fn` anyway &mdash; the downstream code is byte-identical
  for the kernels &mdash; which is exactly why the policy is written down rather
  than assumed.
- `from_f32` widens to `f64` and `to_f32` casts down from `to_f64`, each still
  rounding exactly once.
- **`to_f32` and `to_f64` reassemble the target's bits where the shape fits in
  it.**  A value the target holds exactly needs no arithmetic to decode: NaN,
  an infinity, a subnormal and a normal each take their own branch, and the
  normal branch is a shift, an add and a bit-cast.  The old route &mdash; count
  the subnormal ULPs, convert, scale by `exp2i`, and for `to_f32` cast the
  result down &mdash; is untouched for the shapes that still need it, and still
  rounds exactly once.  The arm is gated on the existing
  `HAS_EXACT_F32_CONVERSION` / `HAS_EXACT_F64_CONVERSION`, so it folds away per
  type.  Min across 15 interleaved passes of alternating builds: the conversion
  route 0.626x, from 0.455x (`F8E4M3FN` division) to 0.806x (`BF16` division),
  with the operator route &mdash; which calls neither conversion &mdash; flat at
  1.006x and its 56 bench bodies verified byte-identical in the symbol table.
  `from_f32` and `from_f64` are unchanged.

  The benchmark's headline moves with the comparator, not the crate: the
  integer route now wins **42 of 56** comparisons at geomean 1.110x, where the
  same binaries measured 56 of 56 at 1.703x before the change.  All fourteen
  losses are additions or subtractions.
- **Subtraction no longer builds a negated operand.**  `Sub` was `self + -rhs`,
  which made a format without a negative zero pay for the guard its negation
  needs &mdash; the code a zero would flip into is its NaN &mdash; on every
  subtraction.  `Add` and `Sub` now share one body that inverts the
  subtrahend's sign where it is read rather than where it is stored, both
  passing a literal flag that folds away.  On an idle box, min across 30
  interleaved passes of alternating builds: `FNUZ` subtraction 0.773&ndash;0.781x,
  addition 0.93&ndash;0.94x, an untouched `mul` control flat at
  0.997&ndash;1.008x.  `FNUZ` subtraction against its own addition goes from
  1.18x to 0.98x, and the three `FNUZ` subtractions a round trip through `f32`
  used to beat are no longer among them.
- The `as` casts in the library now say what they mean.  `cast_signed` and
  `cast_unsigned` &mdash; const-stable and lowering to the same instruction
  &mdash; replace the sign-flipping casts, and `i32::from` replaces the one
  widening cast on a non-const path, so 14 of the 24 `#[allow(clippy::…)]`
  attributes in `src/` are gone rather than suppressed.  What survives is cut to
  the single lint that fires and carries a one-line reason: float-to-int in a
  `const` initialiser, a provably-in-range narrowing where `TryFrom` is not
  const, and the `f64`-to-`f32` cast that *is* the rounding `to_f32` performs.
  The same rewrites went into the test crate where they read better on their own
  terms.
- **`src/lib.rs` now carries `#![warn(clippy::pedantic)]`.**  The suppressions
  above were bookkeeping nothing checked, and 13 of the lints they named had
  already gone stale.  `cargo clippy --all-targets` is the gate now, and a new
  cast that needs an attribute cannot land without one.  The lint is an inner
  attribute, so it is crate-local: dependents and the README doctests see
  nothing.  The test crate is deliberately left out &mdash; pedantic there
  trades `assert_eq!`'s diff for a bare `assert!` and asks for comments
  explaining that a rounding test rounds.

### Removed

- The `NanStyle` enum and `Minifloat::N`, superseded by `Format` and
  `Minifloat::FORMAT`.
- `Minifloat::S`, which was always `true`.  `BITWIDTH` is `1 + E + M`.
- `Minifloat::USE_F32_ADD` and `Minifloat::USE_F32_MUL`, along with their
  inherent twins.  They selected between two routes that were required to
  agree, and the surviving one is no longer the slower.
- The `example` module.
- `F8E3M4FN`, `F8E4M3B11`, and `F8E4M3B11FN`, which are not in LLVM's set.
  They remain one `minifloat!` line away.
- The internal `__conditionally_define_infinities!` macro, folded into the
  format arms of `minifloat!`.

### Fixed

- The OCP MX types decoded their all-ones magnitude as NaN instead of as the
  largest finite value: `F4E2M1FN::MAX` was 4 (now 6), `F6E2M3FN::MAX` was 7
  (now 7.5), and `F6E3M2FN::MAX` was 24 (now 28).
- `F8E4M3FNUZ::MAX` was 480 (now 240) and `F8E5M2FNUZ::MAX` was 114688 (now
  57344), following from the corrected `FNUZ` bias.
- `Finite` types no longer let a NaN input fall through to the rounding path,
  where an all-ones payload could carry out of the exponent field.  `from_f32`
  and `from_f64` saturate to ±`MAX` instead, preserving the sign.
- Conversions no longer read the source exponent field and shift it into place.
  That trick is valid only while the source number is normal *and* the target's
  exponent range sits inside the source's, so it broke wherever a shape reached
  past `f32` or `f64`:
  - `from_f64` returned a nonzero code for `0.0`, and landed two binades off
    for any subnormal `f64`, whenever the type's minimum exponent was below
    `f64`'s.
  - `from_f32` encoded every subnormal `f32` as the maximum value whenever
    counting the target's subnormal ULPs overflowed `f64`.
  - `to_f64` recomputed the *default* exponent bias instead of using the type's
    own, so any type declared with a custom bias decoded wrongly.  Its
    subnormal path flushed to zero one binade too early, and its normal path
    overflowed a `u64` in debug builds.

  Rounding now works on an exact integer significand, which is correct for
  every shape the macro accepts.
- Arithmetic had the same disease, one operation further along: a shape
  reaching past `f64`'s exponent range flushed an operand or a result to zero
  or to infinity on its way through, so `2`<sup>&minus;1000</sup> squared came
  back as zero in a type that represents the answer exactly.  The operators are
  now checked against an exact integer oracle sharing no arithmetic with them,
  exhaustively for every shape up to 8 bits and sampled for the 16-bit ones.

## [0.2.0]

See the git history.
