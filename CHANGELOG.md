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
- Binary `+`, `-`, `*`, `/` always evaluate in `f64` now.  An `f64` operand
  pair carries more than twice the precision any minifloat result needs, so the
  former `f32` shortcut could only ever agree with it.
- `from_f32` widens to `f64` and `to_f32` casts down from `to_f64`, each still
  rounding exactly once.

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

## [0.2.0]

See the git history.
