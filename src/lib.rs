// This file is part of the minifloat project.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod detail;

use core::cmp::Ordering;
use core::f64::consts::LOG10_2;
use core::ops::Neg;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

/// Encoding format of a minifloat
///
/// This is the last argument of [`minifloat!`].  The last three names follow
/// [LLVM/MLIR naming conventions][llvm] derived from their differences to
/// [IEEE 754][ieee].
///
/// [llvm]: https://llvm.org/doxygen/structllvm_1_1APFloatBase.html
/// [ieee]: https://en.wikipedia.org/wiki/IEEE_754
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// Every bit pattern is a finite number
    ///
    /// There are neither infinities nor NaNs, so the all-ones magnitude is the
    /// maximum finite value.  This is the format of the [OCP MX][ocp] sub-8-bit
    /// types.
    ///
    /// [ocp]: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    Finite,

    /// The maximum exponent is reserved for non-finite numbers
    ///
    /// The zero mantissa stands for infinity, while any other value is a NaN.
    IEEE,

    /// Finite with a special NaN encoding
    ///
    /// `F` is for finite, `N` for a special NaN encoding.  There are no
    /// infinities.  The maximum magnitude, exponent and mantissa all ones, is
    /// reserved for NaNs.
    FN,

    /// Finite with a special NaN encoding and unsigned zero
    ///
    /// `F` is for finite, `N` for a special NaN encoding, `UZ` for unsigned
    /// zero.  There are no infinities, and the &minus;0.0 encoding is reserved
    /// for NaN.
    FNUZ,
}

impl Format {
    /// Whether this format has infinities
    ///
    /// This is `true` only for [`IEEE`](Self::IEEE), which reserves the
    /// maximum exponent for them.
    #[must_use]
    #[inline]
    pub const fn has_inf(self) -> bool {
        matches!(self, Self::IEEE)
    }

    /// Whether this format has a NaN encoding
    ///
    /// This is `false` only for [`Finite`](Self::Finite), where every bit
    /// pattern is a number.
    #[must_use]
    #[inline]
    pub const fn has_nan(self) -> bool {
        !matches!(self, Self::Finite)
    }

    /// Whether this format has a negative zero
    ///
    /// This is `false` only for [`FNUZ`](Self::FNUZ), which spends the
    /// would-be &minus;0.0 encoding on NaN.
    #[must_use]
    #[inline]
    pub const fn has_neg_zero(self) -> bool {
        !matches!(self, Self::FNUZ)
    }
}

/// Generic trait for minifloat types
///
/// I am **not** going to implement [`num_traits::Float`][flt] because:
///
/// 1. [`FN`](Format::FN) and [`FNUZ`](Format::FNUZ) types do not have infinities.
/// 2. [`FNUZ`](Format::FNUZ) types do not have a negative zero.
///
/// Binary arithmetic (`+`, `-`, `*`, `/`) is correctly rounded for every shape:
/// each operator works out the result on integer significands, exactly enough
/// to round it once, ties to even.  No hardware float is involved, so a type whose
/// exponent range overruns [`f64`]'s is served as exactly as any other.  An
/// invalid operation — 0/0, ∞ &minus; ∞, ∞ &times; 0, ∞/∞ — yields the format's
/// NaN, or [`MAX`][Self::MAX] where the format has none.
///
/// [flt]: https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html
pub trait Minifloat:
    Copy
    + PartialEq
    + PartialOrd
    + Neg<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    /// Storage type — either [`u8`] or [`u16`]
    ///
    /// A minifloat is a sign bit plus fewer than 16 magnitude bits, so no
    /// wider integer can ever be the right storage.
    type Bits: sealed::Sealed + Copy + Default + Eq + 'static;

    /// Exponent bit-width
    const E: u32;

    /// Significand (mantissa) precision
    const M: u32;

    /// Exponent bias
    const B: i32 = (1 << (Self::E - 1)) - 1;

    /// The encoding format
    const FORMAT: Format;

    /// Whether this format has a negative zero
    ///
    /// This is `false` only for [`FNUZ`](Format::FNUZ) formats, which spend the
    /// would-be &minus;0.0 encoding on NaN.
    const HAS_NEG_ZERO: bool = Self::FORMAT.has_neg_zero();

    /// Total bitwidth
    ///
    /// Equal to 1 + [`E`][Self::E] + [`M`][Self::M]; every minifloat is signed.
    const BITWIDTH: u32 = 1 + Self::E + Self::M;

    /// The radix of the internal representation
    const RADIX: u32 = 2;

    /// The number of digits in the significand, including the implicit leading bit
    ///
    /// Equal to `M` + 1
    const MANTISSA_DIGITS: u32 = Self::M + 1;

    /// The maximum exponent
    ///
    /// Normal numbers < 1 &times; 2<sup>`MAX_EXP`</sup>.
    const MAX_EXP: i32;

    /// One greater than the minimum normal exponent
    ///
    /// Normal numbers ≥ 0.5 &times; 2<sup>`MIN_EXP`</sup>.
    ///
    /// This quirk comes from C macros `FLT_MIN_EXP` and friends.  However, it
    /// is no big deal to mistake it since [[`MIN_POSITIVE`][Self::MIN_POSITIVE],
    /// 2 &times; `MIN_POSITIVE`] is a buffer zone where numbers can be
    /// interpreted as normal or subnormal.
    const MIN_EXP: i32 = 2 - Self::B;

    /// Approximate number of significant decimal digits
    ///
    /// Equal to floor([`M`][Self::M] log<sub>10</sub>(2))
    // A digit count of a minifloat: small and non-negative.  Float-to-int in
    // const has no `try_from`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    const DIGITS: u32 = (Self::M as f64 * crate::LOG10_2) as u32;

    /// Maximum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to floor(log<sub>10</sub>([`MAX`][Self::MAX]))
    const MAX_10_EXP: i32;

    /// Minimum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to ceil(log<sub>10</sub>([`MIN_POSITIVE`][Self::MIN_POSITIVE]))
    // A decimal exponent of a minifloat.  Float-to-int in const has no
    // `try_from`.
    #[allow(clippy::cast_possible_truncation)]
    const MIN_10_EXP: i32 = ((Self::MIN_EXP - 1) as f64 * crate::LOG10_2) as i32;

    /// Whether every value of this type is exactly representable as [`f32`]
    ///
    /// When this is `true`, [`to_f32`][Self::to_f32] is bit-exact and an
    /// `impl From<Self> for f32` is provided.
    const HAS_EXACT_F32_CONVERSION: bool = f32::MANTISSA_DIGITS >= Self::MANTISSA_DIGITS
        && f32::MAX_EXP >= Self::MAX_EXP
        && f32::MIN_EXP <= Self::MIN_EXP;

    /// Whether every value of this type is exactly representable as [`f64`]
    ///
    /// When this is `true`, [`to_f64`][Self::to_f64] is bit-exact and an
    /// `impl From<Self> for f64` is provided.
    const HAS_EXACT_F64_CONVERSION: bool = f64::MANTISSA_DIGITS >= Self::MANTISSA_DIGITS
        && f64::MAX_EXP >= Self::MAX_EXP
        && f64::MIN_EXP <= Self::MIN_EXP;

    /// One representation of NaN, or `None` for formats without one
    ///
    /// A concrete type defines an inherent `NAN` of type `Self` if and only if
    /// this is `Some`.
    const NAN: Option<Self>;

    /// Positive infinity, or `None` for formats without infinities
    ///
    /// A concrete type defines inherent `INFINITY` and `NEG_INFINITY` of type
    /// `Self` if and only if this is `Some`.  There is no separate
    /// `NEG_INFINITY` here because [`Neg`] is a supertrait: negative infinity
    /// is `Self::INFINITY.map(Neg::neg)`.
    const INFINITY: Option<Self>;

    /// The largest number of this type
    ///
    /// This value would be +∞ if the type has infinities.  Otherwise, it is
    /// the maximum finite representation.  This value is also the result of
    /// a positive overflow.
    const HUGE: Self;

    /// The maximum finite number
    const MAX: Self;

    /// The unsigned zero
    ///
    /// This value is useful because we cannot write `0.0` like built-in types.
    const ZERO: Self;

    /// The smallest positive (subnormal) number
    const TINY: Self;

    /// The smallest positive normal number
    ///
    /// Equal to 2<sup>[`MIN_EXP`][Self::MIN_EXP]&minus;1</sup>
    const MIN_POSITIVE: Self;

    /// [Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)
    ///
    /// The difference between 1.0 and the next larger representable number.
    ///
    /// Equal to 2<sup>&minus;`M`</sup>.
    const EPSILON: Self;

    /// The minimum finite number
    ///
    /// Equal to &minus;[`MAX`][Self::MAX]
    const MIN: Self;

    /// Raw transmutation from [`Self::Bits`]
    #[must_use]
    fn from_bits(v: Self::Bits) -> Self;

    /// Raw transmutation to [`Self::Bits`]
    #[must_use]
    fn to_bits(self) -> Self::Bits;

    /// IEEE 754 total-ordering predicate
    ///
    /// The normative definition is lengthy, but it is essentially comparing
    /// sign-magnitude notations.
    ///
    /// The comparison is on the *encoding*, so a NaN sits wherever its format
    /// keeps it: beyond ±[`MAX`][Self::MAX] where it has a sign bit of its own,
    /// but between the negative numbers and +0 for an [`FNUZ`](Format::FNUZ)
    /// type, whose NaN occupies the &minus;0 encoding.
    ///
    /// See also [`f32::total_cmp`],
    /// <https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate>
    #[must_use]
    fn total_cmp(&self, other: &Self) -> Ordering;

    /// Check if the value is NaN
    #[must_use]
    fn is_nan(self) -> bool;

    /// Check if the value is positive or negative infinity
    #[must_use]
    fn is_infinite(self) -> bool;

    /// Check if the value is finite, i.e. neither infinite nor NaN
    #[must_use]
    fn is_finite(self) -> bool;

    /// Check if the value is [subnormal]
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    fn is_subnormal(self) -> bool {
        matches!(self.classify(), core::num::FpCategory::Subnormal)
    }

    /// Check if the value is normal, i.e. not zero, [subnormal], infinite, or NaN
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    fn is_normal(self) -> bool {
        matches!(self.classify(), core::num::FpCategory::Normal)
    }

    /// Classify the value into a floating-point category
    ///
    /// If only one property is going to be tested, it is generally faster to
    /// use the specific predicate instead.
    #[must_use]
    fn classify(self) -> core::num::FpCategory;

    /// Compute the absolute value
    #[must_use]
    fn abs(self) -> Self;

    /// Check if the sign bit is clear
    #[must_use]
    fn is_sign_positive(self) -> bool;

    /// Check if the sign bit is set
    #[must_use]
    fn is_sign_negative(self) -> bool;

    /// Decompose the value into raw `(mantissa, exponent, sign)`
    ///
    /// The triple satisfies `value = sign × mantissa × 2`<sup>`exponent`</sup>
    /// for finite numbers, with the LSB of `mantissa` aligned to the ULP of a
    /// normal number.
    ///
    /// **NaN sentinel**: a NaN value returns `(0, 0, 0)`.  Finite values always
    /// have `sign` equal to ±1, so `sign == 0` unambiguously identifies NaN.
    ///
    /// See also [`f32`'s `integer_decode`][num-traits-integer-decode] in
    /// `num-traits`.
    ///
    /// [num-traits-integer-decode]: https://docs.rs/num-traits/latest/num_traits/float/trait.FloatCore.html#tymethod.integer_decode
    #[must_use]
    fn integer_decode(self) -> (u64, i16, i8);

    /// Probably lossy conversion from [`f32`]
    ///
    /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
    /// Other values are rounded to the nearest representable value.
    #[must_use]
    fn from_f32(x: f32) -> Self;

    /// Probably lossy conversion from [`f64`]
    ///
    /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
    /// Other values are rounded to the nearest representable value.
    #[must_use]
    fn from_f64(x: f64) -> Self;

    /// Best effort conversion to [`f32`]
    #[must_use]
    fn to_f32(self) -> f32;

    /// Best effort conversion to [`f64`]
    #[must_use]
    fn to_f64(self) -> f64;
}

/// Provide a lossless `From<$name> for f32` impl for a minifloat type
///
/// This macro asserts at compile time that
/// [`HAS_EXACT_F32_CONVERSION`][Minifloat::HAS_EXACT_F32_CONVERSION] is `true`.
/// Use it to opt into idiomatic `f32::from(x)` conversion for minifloats whose
/// values are all exactly representable as [`f32`].
///
/// ## Example
///
/// ```
/// use minifloat::{minifloat, impl_from_minifloat_for_f32};
/// minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
/// impl_from_minifloat_for_f32!(F8E4M3FN);
/// let _: f32 = F8E4M3FN::ZERO.into();
/// ```
#[macro_export]
macro_rules! impl_from_minifloat_for_f32 {
    ($name:ty) => {
        const _: () = assert!(
            <$name>::HAS_EXACT_F32_CONVERSION,
            concat!(stringify!($name), " is not exactly representable as f32"),
        );

        impl From<$name> for f32 {
            #[inline]
            fn from(x: $name) -> Self {
                x.to_f32()
            }
        }
    };
}

/// Provide a lossless `From<$name> for f64` impl for a minifloat type
///
/// This macro asserts at compile time that
/// [`HAS_EXACT_F64_CONVERSION`][Minifloat::HAS_EXACT_F64_CONVERSION] is `true`.
/// Use it to opt into idiomatic `f64::from(x)` conversion for minifloats whose
/// values are all exactly representable as [`f64`].
///
/// ## Example
///
/// ```
/// use minifloat::{minifloat, impl_from_minifloat_for_f64};
/// minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
/// impl_from_minifloat_for_f64!(F8E4M3FN);
/// let _: f64 = F8E4M3FN::ZERO.into();
/// ```
#[macro_export]
macro_rules! impl_from_minifloat_for_f64 {
    ($name:ty) => {
        const _: () = assert!(
            <$name>::HAS_EXACT_F64_CONVERSION,
            concat!(stringify!($name), " is not exactly representable as f64"),
        );

        impl From<$name> for f64 {
            #[inline]
            fn from(x: $name) -> Self {
                x.to_f64()
            }
        }
    };
}

/// Configure a (signed) minifloat
///
/// * `$name`: name of the type
/// * `$bits`: the underlying unsigned integer, usually [`u8`] or [`u16`]
/// * `$e`: exponent bit-width
/// * `$m`: explicit significand (mantissa) bit-width
/// * `$b`: exponent bias, which defaults to 2<sup>`$e`&minus;1</sup> &minus; 1
///   for every format but `FNUZ`, whose default is 2<sup>`$e`&minus;1</sup>.
///   Omitting `$n` leaves a bare identifier here ambiguous with a format, so
///   parenthesize a named bias as `(MY_BIAS)`.
/// * `$n`: format, naming a variant of [`Format`]:
///   [`Finite`](Format::Finite), [`IEEE`](Format::IEEE), [`FN`](Format::FN),
///   [`FNUZ`](Format::FNUZ)
///
/// A type declares an inherent `NAN` only if its format has a NaN encoding,
/// and inherent `INFINITY` / `NEG_INFINITY` only if it has infinities.  Generic
/// code reaches them through [`Minifloat::NAN`] and [`Minifloat::INFINITY`],
/// which are [`Option`]s.
///
/// ## Constraints
///
/// * `$e` + `$m` < 16 (there is always a sign bit)
/// * `$e` ≥ 2 (or use an integer type instead)
/// * `$m` > 0 if `$n` is `IEEE` (∞ ≠ NaN)
///
/// ## Example
///
/// ```
/// use minifloat::minifloat;
/// minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
/// ```
#[macro_export]
macro_rules! minifloat {
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, $b:expr, Finite) => {
        $crate::__minifloat!($vis struct $name($bits): $e, $m, $b, Finite);
    };
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, $b:expr, IEEE) => {
        $crate::__minifloat!($vis struct $name($bits): $e, $m, $b, IEEE);

        impl $name {
            /// One representation of NaN
            pub const NAN: Self = Self(Self::NAN_BITS);

            /// Positive infinity
            pub const INFINITY: Self = Self::HUGE;

            /// Negative infinity
            pub const NEG_INFINITY: Self = Self(Self::HUGE.0 | 1 << (Self::E + Self::M));
        }
    };
    // `FN` and `FNUZ`: a NaN encoding, no infinities.  They differ only in the
    // ident they forward, so one arm covers both.
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, $b:expr, $format:ident) => {
        $crate::__minifloat!($vis struct $name($bits): $e, $m, $b, $format);

        impl $name {
            /// One representation of NaN
            pub const NAN: Self = Self(Self::NAN_BITS);
        }
    };
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, FNUZ) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, 1 << ($e - 1), FNUZ);
    };
    // Every other format shares the IEEE default bias.  This arm is why a bare
    // identifier in the third position is a format, not a bias; parenthesize a
    // named bias as `($b)` to reach the arm below.
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, $format:ident) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, (1 << ($e - 1)) - 1, $format);
    };
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, $b:expr) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, $b, IEEE);
    };
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, IEEE);
    };
}

/// Engine behind [`minifloat!`], instantiated by its format arms
///
/// The format arm supplies a [`Format`] variant.  The engine derives the
/// format's properties and special-value bit patterns from it.
#[doc(hidden)]
#[macro_export]
macro_rules! __minifloat {
    ($vis:vis struct $name:ident($bits:ty): $e:expr, $m:expr, $b:expr, $format:ident) => {
        const _: () = assert!($name::BITWIDTH <= <$bits>::BITS);
        const _: () = assert!($name::E >= 2);
        const _: () = assert!($name::M > 0 || !$name::HAS_INF);
        const _: () = assert!($name::MAX_EXP >= 1);
        const _: () = assert!($name::MIN_EXP <= 1);

        // Intra-doc links cannot reach this crate from downstream expansions.
        #[doc = concat!("A minifloat with bit-layout S1E", $e, "M", $m,
            " in the [`", stringify!($format),
            "`](https://docs.rs/minifloat/latest/minifloat/enum.Format.html#variant.",
            stringify!($format), ") format")]
        #[derive(Debug, Clone, Copy, Default)]
        $vis struct $name($bits);

        impl $name {
            /// Exponent bitwidth
            pub const E: u32 = $e;

            /// Explicit significand (mantissa) bitwidth
            ///
            /// This width excludes the implicit leading bit.
            pub const M: u32 = $m;

            /// Exponent bias
            pub const B: i32 = $b;

            /// The encoding format of this type
            pub const FORMAT: $crate::Format = $crate::Format::$format;

            /// Whether this format has infinities
            pub const HAS_INF: bool = Self::FORMAT.has_inf();

            /// Whether this format has a NaN encoding
            pub const HAS_NAN: bool = Self::FORMAT.has_nan();

            /// Whether this format has a negative zero
            pub const HAS_NEG_ZERO: bool = Self::FORMAT.has_neg_zero();

            /// Total bitwidth
            pub const BITWIDTH: u32 = <Self as $crate::Minifloat>::BITWIDTH;

            /// The radix of the internal representation
            pub const RADIX: u32 = <Self as $crate::Minifloat>::RADIX;

            /// The number of digits in the significand, including the implicit leading bit
            ///
            /// Equal to [`M`][Self::M] + 1
            pub const MANTISSA_DIGITS: u32 = <Self as $crate::Minifloat>::MANTISSA_DIGITS;

            /// The maximum exponent
            ///
            /// Normal numbers < 1 &times; 2<sup>`MAX_EXP`</sup>.
            pub const MAX_EXP: i32 = (Self::MAX.0 >> Self::M) as i32 - Self::B + 1;

            /// One greater than the minimum normal exponent
            ///
            /// Normal numbers ≥ 0.5 &times; 2<sup>`MIN_EXP`</sup>.
            ///
            /// This quirk comes from C macros `FLT_MIN_EXP` and friends.  However, it
            /// is no big deal to mistake it since [[`MIN_POSITIVE`][Self::MIN_POSITIVE],
            /// 2 &times; `MIN_POSITIVE`] is a buffer zone where numbers can be
            /// interpreted as normal or subnormal.
            pub const MIN_EXP: i32 = <Self as $crate::Minifloat>::MIN_EXP;

            /// Maximum <var>x</var> such that 10<sup>`x`</sup> is normal
            ///
            /// Equal to floor(log<sub>10</sub>([`MAX`][Self::MAX]))
            // A decimal exponent of a minifloat.  Float-to-int in const has
            // no `try_from`.
            #[allow(clippy::cast_possible_truncation)]
            pub const MAX_10_EXP: i32 = {
                // The significand of `MAX` is one bit shorter when the
                // all-ones magnitude is spent on something else.
                let man_mask = (1 << Self::M) - 1;
                let precision = Self::M + (Self::MAX.0 & man_mask == man_mask) as u32;
                let log2_max = Self::MAX_EXP as f64
                    + $crate::detail::LOG2_SIGNIFICAND[precision as usize];
                (log2_max * core::f64::consts::LOG10_2) as i32
            };

            /// Whether every value of this type is exactly representable as [`f32`]
            ///
            /// When this is `true`, [`to_f32`][Self::to_f32] is bit-exact and an
            /// `impl From<Self> for f32` is provided.
            pub const HAS_EXACT_F32_CONVERSION: bool =
                <Self as $crate::Minifloat>::HAS_EXACT_F32_CONVERSION;

            /// Whether every value of this type is exactly representable as [`f64`]
            ///
            /// When this is `true`, [`to_f64`][Self::to_f64] is bit-exact and an
            /// `impl From<Self> for f64` is provided.
            pub const HAS_EXACT_F64_CONVERSION: bool =
                <Self as $crate::Minifloat>::HAS_EXACT_F64_CONVERSION;

            /// The largest number of this type
            ///
            /// This value would be +∞ if the type has infinities.  Otherwise, it is
            /// the maximum finite representation.  This value is also the result of
            /// a positive overflow.
            pub const HUGE: Self = Self(match Self::FORMAT {
                $crate::Format::IEEE => ((1 << Self::E) - 1) << Self::M,
                $crate::Format::FN => Self::ABS_MASK - 1,
                _ => Self::ABS_MASK, // Finite, FNUZ
            });

            /// The maximum finite number
            pub const MAX: Self = Self(Self::HUGE.0 - Self::HAS_INF as $bits);

            /// The unsigned zero
            pub const ZERO: Self = Self(0);

            /// The smallest positive (subnormal) number
            pub const TINY: Self = Self(1);

            /// The smallest positive normal number
            ///
            /// Equal to 2<sup>[`MIN_EXP`][Self::MIN_EXP]&minus;1</sup>.
            pub const MIN_POSITIVE: Self = Self(1 << Self::M);

            /// [Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)
            ///
            /// The difference between 1.0 and the next larger representable number.
            ///
            /// Equal to 2<sup>&minus;`M`</sup>.
            pub const EPSILON: Self = Self(match Self::B - Self::M.cast_signed() {
                // `s` is a positive exponent of a representable value, so it
                // fits; `TryFrom` is not const.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                s @ 1.. => (s as $bits) << Self::M,
                s => 1 << (Self::M.cast_signed() - 1 + s),
            });

            /// The minimum finite number
            ///
            /// Equal to &minus;[`MAX`][Self::MAX].
            pub const MIN: Self = Self(Self::MAX.0 | 1 << (Self::E + Self::M));

            /// Magnitude mask for internal usage
            const ABS_MASK: $bits = (1 << (Self::E + Self::M)) - 1;

            /// Bit pattern of the canonical NaN
            ///
            /// Meaningful only when [`HAS_NAN`][Self::HAS_NAN] is `true`.
            const NAN_BITS: $bits = match Self::FORMAT {
                $crate::Format::IEEE => ((1 << (Self::E + 1)) - 1) << (Self::M - 1),
                $crate::Format::FN => Self::ABS_MASK,
                $crate::Format::FNUZ => 1 << (Self::E + Self::M),
                _ => 0, // Finite: meaningless, HAS_NAN is false
            };

            #[doc = concat!("Raw transmutation from [`", stringify!($bits), "`]")]
            #[must_use]
            #[inline]
            pub const fn from_bits(v: $bits) -> Self {
                Self(<$bits>::MAX >> (<$bits>::BITS - Self::BITWIDTH) & v)
            }

            #[doc = concat!("Raw transmutation to [`", stringify!($bits), "`]")]
            #[must_use]
            #[inline]
            pub const fn to_bits(self) -> $bits {
                self.0
            }

            /// Check if the value is NaN
            #[must_use]
            // A shape whose `HUGE` is the whole magnitude mask never reaches
            // the `IEEE` arm, where the comparison is then vacuously false.
            #[allow(clippy::bad_bit_mask)]
            #[inline]
            pub const fn is_nan(self) -> bool {
                match Self::FORMAT {
                    $crate::Format::IEEE => self.0 & Self::ABS_MASK > Self::HUGE.0,
                    $crate::Format::FN => self.0 & Self::ABS_MASK == Self::NAN_BITS,
                    $crate::Format::FNUZ => self.0 == Self::NAN_BITS,
                    _ => false, // Finite: there is no NaN
                }
            }

            /// Check if the value is positive or negative infinity
            #[must_use]
            #[inline]
            pub const fn is_infinite(self) -> bool {
                Self::HAS_INF && self.0 & Self::ABS_MASK == Self::HUGE.0
            }

            /// Check if the value is finite, i.e. neither infinite nor NaN
            #[must_use]
            #[inline]
            pub const fn is_finite(self) -> bool {
                !self.is_nan() && !self.is_infinite()
            }

            /// Check if the value is [subnormal]
            ///
            /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
            #[must_use]
            #[inline]
            pub const fn is_subnormal(self) -> bool {
                matches!(self.classify(), core::num::FpCategory::Subnormal)
            }

            /// Check if the value is normal, i.e. not zero, [subnormal], infinite, or NaN
            ///
            /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
            #[must_use]
            #[inline]
            pub const fn is_normal(self) -> bool {
                matches!(self.classify(), core::num::FpCategory::Normal)
            }

            /// Classify the value into a floating-point category
            ///
            /// If only one property is going to be tested, it is generally faster to
            /// use the specific predicate instead.
            #[must_use]
            #[inline]
            pub const fn classify(self) -> core::num::FpCategory {
                if self.is_nan() {
                    core::num::FpCategory::Nan
                } else if self.is_infinite() {
                    core::num::FpCategory::Infinite
                } else {
                    let exp_mask = ((1 << Self::E) - 1) << Self::M;
                    let man_mask = (1 << Self::M) - 1;

                    match (self.0 & exp_mask, self.0 & man_mask) {
                        (0, 0) => core::num::FpCategory::Zero,
                        (0, _) => core::num::FpCategory::Subnormal,
                        (_, _) => core::num::FpCategory::Normal,
                    }
                }
            }

            /// Compute the absolute value
            #[must_use]
            #[inline]
            pub const fn abs(self) -> Self {
                // Without a negative zero, NaN *is* the sign bit; masking it
                // off would turn it into a zero.
                if !Self::HAS_NEG_ZERO && self.0 == Self::NAN_BITS {
                    return Self(Self::NAN_BITS);
                }
                Self::from_bits(self.to_bits() & Self::ABS_MASK)
            }

            /// Check if the sign bit is clear
            #[must_use]
            #[inline]
            pub const fn is_sign_positive(self) -> bool {
                self.0 >> (Self::E + Self::M) & 1 == 0
            }

            /// Check if the sign bit is set
            #[must_use]
            #[inline]
            pub const fn is_sign_negative(self) -> bool {
                self.0 >> (Self::E + Self::M) & 1 == 1
            }

            /// Decompose the value into raw `(mantissa, exponent, sign)`
            ///
            /// The triple satisfies `value = sign × mantissa × 2`<sup>`exponent`</sup>
            /// for finite numbers, with the LSB of `mantissa` aligned to the ULP
            /// of a normal number.
            ///
            /// **NaN sentinel**: a NaN value returns `(0, 0, 0)`.  Finite values
            /// always have `sign` equal to ±1, so `sign == 0` unambiguously
            /// identifies NaN.
            #[must_use]
            // The exponent field is at most 8 bits wide and debiasing it
            // leaves a minifloat exponent.  `TryFrom` is not const.
            #[allow(clippy::cast_possible_truncation)]
            #[inline]
            pub const fn integer_decode(self) -> (u64, i16, i8) {
                if self.is_nan() {
                    return (0, 0, 0);
                }
                let bits = self.to_bits() as u64;
                let sign: i8 = if (bits >> (Self::E + Self::M)) == 0 { 1 } else { -1 };
                let field_mask: u64 = (1u64 << Self::E) - 1;
                let exponent = ((bits >> Self::M) & field_mask) as i32;
                let payload = bits & ((1u64 << Self::M) - 1);
                let mantissa = if exponent == 0 {
                    payload << 1
                } else {
                    payload | (1u64 << Self::M)
                };
                let bias = Self::M.cast_signed() + Self::B;
                (mantissa, (exponent - bias) as i16, sign)
            }

            /// Map sign-magnitude notations to plain unsigned integers
            ///
            /// This serves as a hook for the [`Minifloat`] trait.
            #[inline]
            const fn total_cmp_key(x: $bits) -> $bits {
                let sign = 1 << (Self::E + Self::M);
                let mask = ((x & sign) >> (Self::E + Self::M)) * (sign - 1);
                x ^ (sign | mask)
            }

            /// `const`-callable equality, equivalent to [`PartialEq::eq`]
            ///
            /// `==` cannot be used in `const` contexts on stable Rust until
            /// `const_trait` is stabilized.  This method covers that gap.
            #[must_use]
            #[inline]
            pub const fn const_eq(self, other: Self) -> bool {
                let eq = self.0 == other.0 && !self.is_nan();
                eq || Self::HAS_NEG_ZERO && (self.0 | other.0) & Self::ABS_MASK == 0
            }

            /// `const`-callable partial comparison, equivalent to
            /// [`PartialOrd::partial_cmp`]
            ///
            /// Returns `None` if either operand is NaN.
            #[must_use]
            #[inline]
            pub const fn const_partial_cmp(self, other: Self) -> Option<core::cmp::Ordering> {
                if self.is_nan() || other.is_nan() {
                    return None;
                }
                if self.const_eq(other) {
                    return Some(core::cmp::Ordering::Equal);
                }
                let sign = (self.0 | other.0) >> (Self::E + Self::M) & 1 == 1;
                Some(if (self.0 > other.0) ^ sign {
                    core::cmp::Ordering::Greater
                } else {
                    core::cmp::Ordering::Less
                })
            }

            /// Probably lossy conversion from [`f32`]
            ///
            /// NaNs are preserved where the format has one, and saturate to
            /// ±[`MAX`][Self::MAX] where it does not.  Overflows result in
            /// ±[`HUGE`][Self::HUGE].  Other values are rounded to the nearest
            /// representable value.
            #[must_use]
            #[inline]
            pub fn from_f32(x: f32) -> Self {
                // Widening to `f64` is exact, so this still rounds only once.
                Self::from_f64(f64::from(x))
            }

            /// Probably lossy conversion from [`f64`]
            ///
            /// NaNs are preserved where the format has one, and saturate to
            /// ±[`MAX`][Self::MAX] where it does not.  Overflows result in
            /// ±[`HUGE`][Self::HUGE].  Other values are rounded to the nearest
            /// representable value.
            #[must_use]
            #[inline]
            pub fn from_f64(x: f64) -> Self {
                let sign_bit = <$bits>::from(x.is_sign_negative()) << (Self::E + Self::M);

                if x.is_nan() {
                    let magnitude = if Self::HAS_NAN { Self::NAN_BITS } else { Self::MAX.0 };
                    return Self::from_bits(magnitude | sign_bit);
                }

                let (significand, exponent) = $crate::detail::decompose(x.abs());
                Self::from_parts(x.is_sign_negative(), significand, exponent)
            }

            /// Correctly rounded `sign` &times; `significand` &times;
            /// 2<sup>`exponent`</sup>
            ///
            /// This is where every rounding in the crate happens.  The triple
            /// is an exact number, or one whose lowest bit is sticky, so it
            /// can come from an [`f64`] or from arithmetic of its own.
            #[inline]
            fn from_parts(negative: bool, significand: u64, exponent: i32) -> Self {
                let sign_bit = <$bits>::from(negative) << (Self::E + Self::M);

                if significand == 0 {
                    // Without a negative zero, signing a zero spells NaN.
                    return Self::from_bits(<$bits>::from(Self::HAS_NEG_ZERO) * sign_bit);
                }

                // The exponent of the value, which is in [2^e, 2^(e+1)).
                let e = exponent + (u64::BITS - 1 - significand.leading_zeros()).cast_signed();

                let magnitude = if e < Self::MIN_EXP - 1 {
                    // Subnormal numbers all share the ULP of the smallest one,
                    // so their code *is* the rounded multiple of that ULP.
                    $crate::detail::round_to_scale(
                        significand, exponent, Self::MIN_EXP - 1 - Self::M.cast_signed())
                } else {
                    // Rounding to `MANTISSA_DIGITS` may carry into the implicit
                    // bit.  That lands on the next exponent field with a zero
                    // mantissa, which is exactly where the extra ULP belongs.
                    let rounded = $crate::detail::round_to_scale(
                        significand, exponent, e - Self::M.cast_signed());
                    (i64::from(e + Self::B) << Self::M) + rounded - (1 << Self::M)
                };

                // Clamped to `HUGE` just above, so it fits and is non-negative;
                // `TryFrom` is not const-friendly here either.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let magnitude = magnitude.min(i64::from(Self::HUGE.0)) as $bits;
                // A value that rounds to zero drops its sign for the same
                // reason a zero does.
                let signed = <$bits>::from(Self::HAS_NEG_ZERO || magnitude != 0);
                Self::from_bits(magnitude | (signed * sign_bit))
            }

            /// Split a finite value into exact `(sign, significand, exponent)`
            ///
            /// The inverse of [`from_parts`][Self::from_parts] where the value
            /// is representable: `significand` carries the implicit bit where
            /// the code has one, and `exponent` is the ULP scale of the code.
            #[inline]
            const fn to_parts(self) -> (bool, u64, i32) {
                let magnitude = (self.0 & Self::ABS_MASK) as u64;
                let field = (magnitude >> Self::M) as i32;
                let fraction = magnitude & ((1 << Self::M) - 1);

                if field == 0 {
                    (self.is_sign_negative(), fraction, 1 - Self::B - Self::M.cast_signed())
                } else {
                    (
                        self.is_sign_negative(),
                        fraction | 1 << Self::M,
                        field - Self::B - Self::M.cast_signed(),
                    )
                }
            }

            /// `self` + `rhs`, or `self` &minus; `rhs` when `flip` is set
            ///
            /// Subtraction is addition with the subtrahend's sign flipped.
            /// Flipping it here, in the one place that reads a sign, is what
            /// spares a format without a negative zero the guard its `Neg`
            /// needs: there is no intermediate value to keep representable,
            /// only a bool to invert.  Both callers pass a literal, so the
            /// flag folds away before anything is emitted.
            #[inline]
            fn add_impl(self, rhs: Self, flip: bool) -> Self {
                if self.is_nan() || rhs.is_nan() {
                    return Self::INVALID;
                }
                if self.is_infinite() && rhs.is_infinite() {
                    // Two infinities agree only when their signs do; what the
                    // other case ought to be is exactly the question.
                    let agree = self.is_sign_negative() == (rhs.is_sign_negative() ^ flip);
                    return if agree { self } else { Self::INVALID };
                }
                // An infinity outweighs anything finite added to it.
                if self.is_infinite() {
                    return self;
                }
                if rhs.is_infinite() {
                    // Reachable only where the format has infinities, and
                    // there a negation is the bare XOR it looks like.
                    return if flip { -rhs } else { rhs };
                }
                let (negative, significand, exponent) = self.to_parts();
                let (rhs_negative, rhs_significand, rhs_exponent) = rhs.to_parts();
                let (negative, significand, exponent) = $crate::detail::add_parts(
                    negative, significand, exponent,
                    rhs_negative ^ flip, rhs_significand, rhs_exponent);
                Self::from_parts(negative, significand, exponent)
            }

            /// The result of an invalid operation
            ///
            /// A format without a NaN saturates one to `MAX`, as
            /// [`from_f64`][Self::from_f64] does for a NaN input.  The sign of
            /// a default NaN means nothing, so this one is positive.
            const INVALID: Self = Self(if Self::HAS_NAN { Self::NAN_BITS } else { Self::MAX.0 });

            /// [`HUGE`][Self::HUGE] with the sign an operation worked out
            #[inline]
            const fn huge(negative: bool) -> Self {
                Self(Self::HUGE.0 | ((negative as $bits) << (Self::E + Self::M)))
            }

            /// Best effort conversion to [`f64`]
            ///
            /// This is exact unless the exponent leaves [`f64`]'s range, where
            /// the result rounds, overflows, or underflows like any other
            /// [`f64`] computation.
            #[must_use]
            #[inline]
            pub fn to_f64(self) -> f64 {
                let sign = if self.is_sign_negative() { -1.0 } else { 1.0 };

                if self.is_nan() {
                    return f64::NAN.copysign(sign);
                }
                if self.is_infinite() {
                    return f64::INFINITY * sign;
                }

                let magnitude = self.to_bits() & Self::ABS_MASK;
                let field = magnitude >> Self::M;

                if Self::HAS_EXACT_F64_CONVERSION {
                    // A shape that fits needs no rounding at all: a subnormal is
                    // its code times the ULP, exactly, and a normal value is its
                    // own bits with the exponent field rebiased.  Every
                    // expression here stays total for the shapes that never take
                    // this branch, whose rebias is negative.
                    if field == 0 {
                        let scale = $crate::detail::exp2i(
                            Self::MIN_EXP - Self::MANTISSA_DIGITS.cast_signed());
                        return f64::from(magnitude) * scale.copysign(sign);
                    }
                    let shifted =
                        u64::from(magnitude) << (f64::MANTISSA_DIGITS - Self::MANTISSA_DIGITS);
                    let rebias =
                        ((Self::MIN_EXP - f64::MIN_EXP) as u64) << (f64::MANTISSA_DIGITS - 1);
                    let sign_bit = u64::from(self.is_sign_negative()) << (u64::BITS - 1);
                    return f64::from_bits(sign_bit | (shifted + rebias));
                }

                let (significand, exponent) = if field == 0 {
                    (magnitude, 1 - Self::B - Self::M.cast_signed())
                } else {
                    (
                        (magnitude & ((1 << Self::M) - 1)) | 1 << Self::M,
                        i32::from(field) - Self::B - Self::M.cast_signed(),
                    )
                };

                // Splitting the scale keeps either factor inside `f64`'s
                // exponent range, so the first product is exact and the second
                // rounds at most once.  A single `exp2i` would flush to zero or
                // to infinity long before the product does.
                let head = exponent.clamp(f64::MIN_EXP - 1, f64::MAX_EXP - 1);
                sign * f64::from(significand)
                    * $crate::detail::exp2i(head)
                    * $crate::detail::exp2i(exponent - head)
            }

            /// Best effort conversion to [`f32`]
            #[must_use]
            #[inline]
            pub fn to_f32(self) -> f32 {
                let sign = if self.is_sign_negative() { -1.0 } else { 1.0 };

                if self.is_nan() {
                    // A narrowing cast is not specified to keep the NaN sign.
                    return f32::NAN.copysign(sign);
                }

                if Self::HAS_EXACT_F32_CONVERSION {
                    // Same reassembly as `to_f64`, one size down; see there for
                    // why nothing rounds and nothing can fail for a shape that
                    // skips this.
                    if self.is_infinite() {
                        return f32::INFINITY * sign;
                    }
                    let magnitude = self.to_bits() & Self::ABS_MASK;
                    if magnitude >> Self::M == 0 {
                        // The scale of a shape that fits in `f32` is exact in
                        // `f32`; the cast only drops `f64`'s spare exponent room.
                        #[allow(clippy::cast_possible_truncation)]
                        let scale = $crate::detail::exp2i(
                            Self::MIN_EXP - Self::MANTISSA_DIGITS.cast_signed(),
                        ) as f32;
                        return f32::from(magnitude) * scale.copysign(sign);
                    }
                    let shifted =
                        u32::from(magnitude) << (f32::MANTISSA_DIGITS - Self::MANTISSA_DIGITS);
                    let rebias =
                        ((Self::MIN_EXP - f32::MIN_EXP) as u32) << (f32::MANTISSA_DIGITS - 1);
                    let sign_bit = u32::from(self.is_sign_negative()) << (u32::BITS - 1);
                    return f32::from_bits(sign_bit | (shifted + rebias));
                }

                // Conversion to `f64` is exact but for exponents out of its
                // range, which are out of `f32`'s range too.  Either way the
                // cast below is the only rounding.
                #[allow(clippy::cast_possible_truncation)]
                { self.to_f64() as f32 }
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.const_eq(*other)
            }
        }

        impl core::hash::Hash for $name {
            /// Hash a minifloat by its bit pattern, normalizing zero so `+0` and
            /// `-0` hash to the same value where the format has both.  NaN
            /// values hash by their raw bits, consistent with `NaN != NaN`.
            #[inline]
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                let bits = if !self.is_nan() && self.0 & Self::ABS_MASK == 0 {
                    <$bits>::default()
                } else {
                    self.0
                };
                bits.hash(state);
            }
        }

        impl PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.const_partial_cmp(*other)
            }
        }

        impl core::ops::Neg for $name {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                let sign = 1 << (Self::E + Self::M);
                let magnitude = self.0 & Self::ABS_MASK;
                // Without a negative zero, a zero has no sign to flip: the
                // code it would flip into is the NaN.  The top bit of
                // `m | -m` says whether the magnitude is nonzero; `m == 0`
                // asks the same in a `setcc`, whose partial write of a byte
                // register carries a false dependency on whatever was last
                // in it.  The caller that made that stall visible was `sub`,
                // which no longer comes through here -- it inverts a sign
                // flag instead.  This stays as the cheaper way to ask.
                let switch = if Self::HAS_NEG_ZERO {
                    sign
                } else {
                    (magnitude | magnitude.wrapping_neg()) & sign
                };
                Self(self.0 ^ switch)
            }
        }

        impl core::ops::Add for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                self.add_impl(rhs, false)
            }
        }

        impl core::ops::Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.add_impl(rhs, true)
            }
        }

        impl core::ops::Mul for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                if self.is_nan() || rhs.is_nan() {
                    return Self::INVALID;
                }
                let negative = self.is_sign_negative() != rhs.is_sign_negative();
                let (_, significand, exponent) = self.to_parts();
                let (_, rhs_significand, rhs_exponent) = rhs.to_parts();

                if self.is_infinite() || rhs.is_infinite() {
                    // An infinity scaled by zero is the invalid one; the
                    // significands say which operand is the zero.
                    let zero = significand == 0 || rhs_significand == 0;
                    return if zero { Self::INVALID } else { Self::huge(negative) };
                }
                // Two significands of at most 15 bits multiply exactly.
                Self::from_parts(negative, significand * rhs_significand, exponent + rhs_exponent)
            }
        }

        impl core::ops::Div for $name {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                if self.is_nan() || rhs.is_nan() {
                    return Self::INVALID;
                }
                let negative = self.is_sign_negative() != rhs.is_sign_negative();
                let (_, significand, exponent) = self.to_parts();
                let (_, rhs_significand, rhs_exponent) = rhs.to_parts();

                if self.is_infinite() {
                    // Infinity over infinity is the invalid one.
                    return if rhs.is_infinite() { Self::INVALID } else { Self::huge(negative) };
                }
                if rhs.is_infinite() {
                    return Self::from_parts(negative, 0, 0);
                }
                if rhs_significand == 0 {
                    // Zero over zero is the invalid one; anything else over
                    // zero overflows every exponent there is.
                    return if significand == 0 { Self::INVALID } else { Self::huge(negative) };
                }
                let (significand, exponent) = $crate::detail::div_parts(
                    significand, exponent, rhs_significand, rhs_exponent);
                Self::from_parts(negative, significand, exponent)
            }
        }

        impl core::ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
        }

        impl core::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
        }

        impl core::ops::MulAssign for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
        }

        impl core::ops::DivAssign for $name {
            #[inline]
            fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
        }

        impl $crate::Minifloat for $name {
            type Bits = $bits;
            const E: u32 = $name::E;
            const M: u32 = $name::M;
            const B: i32 = $name::B;
            const FORMAT: $crate::Format = $name::FORMAT;
            const MAX_EXP: i32 = $name::MAX_EXP;
            const MAX_10_EXP: i32 = $name::MAX_10_EXP;

            const NAN: Option<Self> =
                if Self::HAS_NAN { Some(Self(Self::NAN_BITS)) } else { None };
            const INFINITY: Option<Self> =
                if Self::HAS_INF { Some($name::HUGE) } else { None };
            const HUGE: Self = Self::HUGE;
            const MAX: Self = Self::MAX;

            const ZERO: Self = Self::ZERO;
            const TINY: Self = Self::TINY;
            const MIN_POSITIVE: Self = Self::MIN_POSITIVE;
            const EPSILON: Self = Self::EPSILON;
            const MIN: Self = Self::MIN;

            #[inline]
            fn from_bits(v: Self::Bits) -> Self {
                Self::from_bits(v)
            }

            #[inline]
            fn to_bits(self) -> Self::Bits {
                self.to_bits()
            }

            #[inline]
            fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                Self::total_cmp_key(self.0).cmp(&Self::total_cmp_key(other.0))
            }

            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }

            #[inline]
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            #[inline]
            fn is_finite(self) -> bool {
                self.is_finite()
            }

            #[inline]
            fn classify(self) -> core::num::FpCategory {
                self.classify()
            }

            #[inline]
            fn abs(self) -> Self {
                self.abs()
            }

            #[inline]
            fn is_sign_positive(self) -> bool {
                self.is_sign_positive()
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }

            #[inline]
            fn integer_decode(self) -> (u64, i16, i8) {
                self.integer_decode()
            }

            #[inline]
            fn from_f32(x: f32) -> Self {
                Self::from_f32(x)
            }

            #[inline]
            fn from_f64(x: f64) -> Self {
                Self::from_f64(x)
            }

            #[inline]
            fn to_f32(self) -> f32 {
                Self::to_f32(self)
            }

            #[inline]
            fn to_f64(self) -> f64 {
                Self::to_f64(self)
            }
        }
    };
}

/// Opt into the lossless `From` impls for both [`f32`] and [`f64`]
macro_rules! lossless {
    ($($name:ident)*) => {$(
        impl_from_minifloat_for_f32!($name);
        impl_from_minifloat_for_f64!($name);
    )*};
}

// LLVM's set of floating-point types no wider than 16 bits.  Following LLVM,
// the OCP MX types keep their historical `FN` suffix even though their format
// is `Finite` — every bit pattern is a finite number, so the all-ones
// magnitude is the maximum value rather than a NaN.  Each generated type
// documents its own format.
minifloat!(pub struct F4E2M1FN(u8): 2, 1, Finite);
minifloat!(pub struct F6E2M3FN(u8): 2, 3, Finite);
minifloat!(pub struct F6E3M2FN(u8): 3, 2, Finite);
minifloat!(pub struct F8E3M4(u8): 3, 4);
minifloat!(pub struct F8E4M3(u8): 4, 3);
minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
minifloat!(pub struct F8E4M3FNUZ(u8): 4, 3, FNUZ);
minifloat!(pub struct F8E4M3B11FNUZ(u8): 4, 3, 11, FNUZ);
minifloat!(pub struct F8E5M2(u8): 5, 2);
minifloat!(pub struct F8E5M2FNUZ(u8): 5, 2, FNUZ);
minifloat!(pub struct F16(u16): 5, 10);
minifloat!(pub struct BF16(u16): 8, 7);

// Every type above is narrow enough to be exact in both `f32` and `f64`.
lossless! {
    F4E2M1FN F6E2M3FN F6E3M2FN
    F8E3M4 F8E4M3 F8E4M3FN F8E4M3FNUZ F8E4M3B11FNUZ F8E5M2 F8E5M2FNUZ
    F16 BF16
}
