// This file is part of the minifloat project.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod detail;
pub mod example;

use core::cmp::Ordering;
use core::f64::consts::LOG10_2;
use core::ops::Neg;

/// NaN encoding style
///
/// The variants follow [LLVM/MLIR naming conventions][llvm] derived from
/// their differences to [IEEE 754][ieee].
///
/// [llvm]: https://llvm.org/doxygen/structllvm_1_1APFloatBase.html
/// [ieee]: https://en.wikipedia.org/wiki/IEEE_754
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanStyle {
    /// IEEE 754 NaN encoding
    ///
    /// The maximum exponent is reserved for non-finite numbers.  The zero
    /// mantissa stands for infinity, while any other value represents a NaN.
    IEEE,

    /// `FN` suffix as in LLVM/MLIR
    ///
    /// `F` is for finite, `N` for a special NaN encoding.  There are no
    /// infinities.  The maximum magnitude is reserved for NaNs, where the
    /// exponent and mantissa are all ones.
    FN,

    /// `FNUZ` suffix as in LLVM/MLIR
    ///
    /// `F` is for finite, `N` for a special NaN encoding, `UZ` for unsigned
    /// zero.  There are no infinities.  The negative zero (&minus;0.0)
    /// representation is reserved for NaN.  As a result, there is only one
    /// (+0.0) unsigned zero.
    FNUZ,
}

#[allow(clippy::excessive_precision)]
const LOG2_SIGNIFICAND: [f64; 16] = [
    -2.0,
    -1.0,
    -4.150_374_992_788_438_13e-1,
    -1.926_450_779_423_958_81e-1,
    -9.310_940_439_148_146_51e-2,
    -4.580_368_961_312_478_86e-2,
    -2.272_007_650_008_352_89e-2,
    -1.131_531_322_783_414_61e-2,
    -5.646_563_141_142_062_72e-3,
    -2.820_519_062_378_662_63e-3,
    -1.409_570_254_671_353_63e-3,
    -7.046_129_765_893_727_06e-4,
    -3.522_634_716_290_213_85e-4,
    -1.761_209_842_740_240_62e-4,
    -8.805_780_458_002_638_34e-5,
    -4.402_823_044_177_721_15e-5,
];

pub trait Minifloat: Copy + PartialEq + PartialOrd + Neg<Output = Self> {
    /// Storage type
    type Bits;

    /// Whether the type is signed
    const S: bool = true;

    /// Exponent bit-width
    const E: u32;

    /// Significand (mantissa) precision
    const M: u32;

    /// Exponent bias
    const B: i32 = (1 << (Self::E - 1)) - 1;

    /// NaN encoding style
    const N: NanStyle = NanStyle::IEEE;

    /// Total bitwidth
    const BITWIDTH: u32 = Self::S as u32 + Self::E + Self::M;

    /// The radix of the internal representation
    const RADIX: u32 = 2;

    /// The number of digits in the significand, including the implicit leading bit
    ///
    /// Equal to `M` + 1
    const MANTISSA_DIGITS: u32 = Self::M + 1;

    /// The maximum exponent
    ///
    /// Normal numbers < 1 &times; 2<sup>`MAX_EXP`</sup>.
    const MAX_EXP: i32 = (1 << Self::E)
        - Self::B
        - match Self::N {
            NanStyle::IEEE => 1,
            NanStyle::FN => (Self::M == 0) as i32,
            NanStyle::FNUZ => 0,
        };

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
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    const DIGITS: u32 = (Self::M as f64 * crate::LOG10_2) as u32;

    /// Maximum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to floor(log<sub>10</sub>([`MAX`][Self::MAX]))
    #[allow(clippy::cast_possible_truncation)]
    const MAX_10_EXP: i32 = {
        let exponent = (1 << Self::E) - Self::B - matches!(Self::N, NanStyle::IEEE) as i32;
        let precision = Self::M + !matches!(Self::N, NanStyle::FN) as u32;
        let log2_max = exponent as f64 + crate::LOG2_SIGNIFICAND[precision as usize];
        (log2_max * crate::LOG10_2) as i32
    };

    /// Minimum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to ceil(log<sub>10</sub>([`MIN_POSITIVE`][Self::MIN_POSITIVE]))
    #[allow(clippy::cast_possible_truncation)]
    const MIN_10_EXP: i32 = ((Self::MIN_EXP - 1) as f64 * crate::LOG10_2) as i32;

    /// One representation of NaN
    const NAN: Self;

    /// The largest number of this type
    ///
    /// This value would be +∞ if the type has infinities.  Otherwise, it is
    /// the maximum finite representation.  This value is also the result of
    /// a positive overflow.
    const HUGE: Self;

    /// The maximum finite number
    const MAX: Self;

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
}

/// Internal macro to conditionally define infinities
#[doc(hidden)]
#[macro_export]
macro_rules! __conditionally_define_infinities {
    (impl $name:ident, IEEE) => {
        impl $name {
            /// Positive infinity
            pub const INFINITY: Self = Self::HUGE;

            /// Negative infinity
            pub const NEG_INFINITY: Self = Self(Self::HUGE.0 | (1 << (Self::E + Self::M)));
        }
    };
    (impl $name:ident, $n:ident) => {};
}

/// Internal macro to select the correct sized trait implementation
///
/// This macro needs to be public for [`minifloat!`] to invoke, but it is not
/// intended for general use.
#[doc(hidden)]
#[macro_export]
macro_rules! __select_sized_trait {
    (u8, $name:ident, $e:expr, $m:expr) => {
        impl $crate::Most8<$m> for $name {
            const E: u32 = Self::E;
            const B: i32 = Self::B;
            const N: $crate::NanStyle = Self::N;

            const NAN: Self = Self::NAN;
            const HUGE: Self = Self::HUGE;
            const MAX: Self = Self::MAX;
            const TINY: Self = Self::TINY;
            const MIN_POSITIVE: Self = Self::MIN_POSITIVE;
            const EPSILON: Self = Self::EPSILON;
            const MIN: Self = Self::MIN;

            fn from_bits(v: u8) -> Self {
                Self::from_bits(v)
            }

            fn to_bits(self) -> u8 {
                self.to_bits()
            }

            fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                Self::total_cmp_key(self.0).cmp(&Self::total_cmp_key(other.0))
            }
        }
    };
    (u16, $name:ident, $e:expr, $m:expr) => {
        impl $crate::Most16<$m> for $name {
            const E: u32 = Self::E;
            const B: i32 = Self::B;
            const N: $crate::NanStyle = Self::N;

            const NAN: Self = Self::NAN;
            const HUGE: Self = Self::HUGE;
            const MAX: Self = Self::MAX;
            const TINY: Self = Self::TINY;
            const MIN_POSITIVE: Self = Self::MIN_POSITIVE;
            const EPSILON: Self = Self::EPSILON;
            const MIN: Self = Self::MIN;

            fn from_bits(v: u16) -> Self {
                Self::from_bits(v)
            }

            fn to_bits(self) -> u16 {
                self.to_bits()
            }

            fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                Self::total_cmp_key(self.0).cmp(&Self::total_cmp_key(other.0))
            }
        }
    };
}

/// Define a minifloat taking up to 16 bits
///
/// * `$name`: name of the type
/// * `$bits`: the underlying integer type, which must be [`u8`] or [`u16`]
/// * `$e`: exponent bit-width
/// * `$m`: explicit significand (mantissa) bit-width
/// * `$b`: exponent bias, which defaults to 2<sup>`$e`&minus;1</sup> &minus; 1
/// * `$n`: NaN encoding style, one of the [`NanStyle`] variants
///
/// ## Constraints
///
/// * `$e` + `$m` < 16 (there is always a sign bit)
/// * `$e` ≥ 2 (or use an integer type instead)
/// * `$m` > 0 if `$n` is [`IEEE`][NanStyle::IEEE] (∞ ≠ NaN)
///
/// ## Example
///
/// ```
/// use minifloat::minifloat;
/// minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
/// ```
#[macro_export]
macro_rules! minifloat {
    ($vis:vis struct $name:ident($bits:tt): $e:expr, $m:expr, $b:expr, $n:ident) => {
        #[allow(non_camel_case_types)]
        #[doc = concat!("A minifloat with bit-layout S1E", $e, "M", $m)]
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

            /// NaN encoding style
            pub const N: $crate::NanStyle = $crate::NanStyle::$n;

            /// Total bitwidth
            pub const BITWIDTH: u32 = 1 + Self::E + Self::M;

            /// The radix of the internal representation
            pub const RADIX: u32 = 2;

            /// The number of digits in the significand, including the implicit leading bit
            ///
            /// Equal to [`M`][Self::M] + 1
            pub const MANTISSA_DIGITS: u32 = $m + 1;

            /// The maximum exponent
            ///
            /// Normal numbers < 1 &times; 2<sup>`MAX_EXP`</sup>.
            pub const MAX_EXP: i32 = (1 << Self::E)
                - Self::B
                - match Self::N {
                    $crate::NanStyle::IEEE => 1,
                    $crate::NanStyle::FN => (Self::M == 0) as i32,
                    $crate::NanStyle::FNUZ => 0,
                };

            /// One greater than the minimum normal exponent
            ///
            /// Normal numbers ≥ 0.5 &times; 2<sup>`MIN_EXP`</sup>.
            ///
            /// This quirk comes from C macros `FLT_MIN_EXP` and friends.  However, it
            /// is no big deal to mistake it since [[`MIN_POSITIVE`][Self::MIN_POSITIVE],
            /// 2 &times; `MIN_POSITIVE`] is a buffer zone where numbers can be
            /// interpreted as normal or subnormal.
            pub const MIN_EXP: i32 = 2 - Self::B;

            /// One representation of NaN
            pub const NAN: Self = Self(match Self::N {
                $crate::NanStyle::IEEE => ((1 << (Self::E + 1)) - 1) << (Self::M - 1),
                $crate::NanStyle::FN => (1 << (Self::E + Self::M)) - 1,
                $crate::NanStyle::FNUZ => 1 << (Self::E + Self::M),
            });

            /// The largest number of this type
            ///
            /// This value would be +∞ if the type has infinities.  Otherwise, it is
            /// the maximum finite representation.  This value is also the result of
            /// a positive overflow.
            pub const HUGE: Self = Self(match Self::N {
                $crate::NanStyle::IEEE => ((1 << Self::E) - 1) << Self::M,
                $crate::NanStyle::FN => (1 << (Self::E + Self::M)) - 2,
                $crate::NanStyle::FNUZ => (1 << (Self::E + Self::M)) - 1,
            });

            /// The maximum finite number
            pub const MAX: Self = Self(Self::HUGE.0 - matches!(Self::N, $crate::NanStyle::IEEE) as $bits);

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
            #[allow(clippy::cast_possible_wrap)]
            pub const EPSILON: Self = Self(match Self::B - Self::M as i32 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                s @ 1.. => (s as $bits) << Self::M,
                s => 1 << (Self::M as i32 - 1 + s),
            });

            /// The minimum finite number
            ///
            /// Equal to &minus;[`MAX`][Self::MAX].
            pub const MIN: Self = Self(Self::MAX.0 | 1 << (Self::E + Self::M));

            /// Magnitude mask for internal usage
            const ABS_MASK: $bits = (1 << (Self::E + Self::M)) - 1;

            #[doc = concat!("Raw transmutation from [`", stringify!($bits), "`]")]
            #[must_use]
            pub const fn from_bits(v: $bits) -> Self {
                Self($bits::MAX >> ($bits::BITS - Self::BITWIDTH) & v)
            }

            #[doc = concat!("Raw transmutation to [`", stringify!($bits), "`]")]
            #[must_use]
            pub const fn to_bits(self) -> $bits {
                self.0
            }

            /// Check if the value is NaN
            #[must_use]
            pub const fn is_nan(self) -> bool {
                match Self::N {
                    #[allow(clippy::bad_bit_mask)]
                    $crate::NanStyle::IEEE => self.0 & Self::ABS_MASK > Self::HUGE.0,
                    $crate::NanStyle::FN => self.0 & Self::ABS_MASK == Self::NAN.0 & Self::ABS_MASK,
                    $crate::NanStyle::FNUZ => self.0 == Self::NAN.0,
                }
            }

            /// Check if the value is positive or negative infinity
            #[must_use]
            pub const fn is_infinite(self) -> bool {
                matches!(Self::N, $crate::NanStyle::IEEE) && self.0 & Self::ABS_MASK == Self::HUGE.0
            }

            /// Check if the value is finite, i.e. neither infinite nor NaN
            #[must_use]
            pub const fn is_finite(self) -> bool {
                match Self::N {
                    $crate::NanStyle::IEEE => self.0 & Self::ABS_MASK < Self::HUGE.0,
                    _ => !self.is_nan(),
                }
            }

            /// Check if the value is [subnormal]
            ///
            /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
            #[must_use]
            pub const fn is_subnormal(self) -> bool {
                matches!(self.classify(), core::num::FpCategory::Subnormal)
            }

            /// Check if the value is normal, i.e. not zero, [subnormal], infinite, or NaN
            ///
            /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
            #[must_use]
            pub const fn is_normal(self) -> bool {
                matches!(self.classify(), core::num::FpCategory::Normal)
            }

            /// Classify the value into a floating-point category
            ///
            /// If only one property is going to be tested, it is generally faster to
            /// use the specific predicate instead.
            #[must_use]
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
            pub const fn abs(self) -> Self {
                if matches!(Self::N, $crate::NanStyle::FNUZ) && self.0 == Self::NAN.0 {
                    return Self::NAN;
                }
                Self::from_bits(self.to_bits() & Self::ABS_MASK)
            }

            /// Check if the sign bit is clear
            #[must_use]
            pub const fn is_sign_positive(self) -> bool {
                self.0 >> (Self::E + Self::M) & 1 == 0
            }

            /// Check if the sign bit is set
            #[must_use]
            pub const fn is_sign_negative(self) -> bool {
                self.0 >> (Self::E + Self::M) & 1 == 1
            }

            /// Map sign-magnitude notations to plain unsigned integers
            ///
            /// This serves as a hook for the [`Minifloat`] trait.
            const fn total_cmp_key(x: $bits) -> $bits {
                let sign = 1 << (Self::E + Self::M);
                let mask = ((x & sign) >> (Self::E + Self::M)) * (sign - 1);
                x ^ (sign | mask)
            }

            /// Probably lossy conversion from [`f32`]
            ///
            /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
            /// Other values are rounded to the nearest representable value.
            #[must_use]
            #[allow(clippy::cast_possible_wrap)]
            pub fn from_f32(x: f32) -> Self {
                if x.is_nan() {
                    let sign_bit = <$bits>::from(x.is_sign_negative()) << (Self::E + Self::M);
                    return Self::from_bits(Self::NAN.0 | sign_bit);
                }

                let bits = $crate::detail::round_f32_to_precision::<$m>(x).to_bits();
                let sign_bit = ((bits >> 31) as $bits) << (Self::E + Self::M);
                let diff = (Self::MIN_EXP - f32::MIN_EXP) << Self::M;
                let magnitude = bits << 1 >> (f32::MANTISSA_DIGITS - Self::M);
                let magnitude = magnitude as i32 - diff;

                if magnitude < 1 << Self::M {
                    let ticks = f64::from(x.abs()) * $crate::detail::exp2i(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let ticks = ticks.round_ties_even() as $bits;
                    return Self::from_bits(
                        (<$bits>::from(Self::N != $crate::NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks,
                    );
                }

                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Self::from_bits(magnitude.min(i32::from(Self::HUGE.to_bits())) as $bits | sign_bit)
            }

            /// Probably lossy conversion from [`f64`]
            ///
            /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
            /// Other values are rounded to the nearest representable value.
            #[must_use]
            #[allow(clippy::cast_possible_wrap)]
            pub fn from_f64(x: f64) -> Self {
                if x.is_nan() {
                    let sign_bit = <$bits>::from(x.is_sign_negative()) << (Self::E + Self::M);
                    return Self::from_bits(Self::NAN.to_bits() | sign_bit);
                }

                let bits = $crate::detail::round_f64_to_precision::<$m>(x).to_bits();
                let sign_bit = ((bits >> 63) as $bits) << (Self::E + Self::M);
                let diff = i64::from(Self::MIN_EXP - f64::MIN_EXP) << Self::M;
                let magnitude = bits << 1 >> (f64::MANTISSA_DIGITS - Self::M);
                let magnitude = magnitude as i64 - diff;

                if magnitude < 1 << Self::M {
                    let ticks = x.abs() * $crate::detail::exp2i(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let ticks = ticks.round_ties_even() as $bits;
                    return Self::from_bits(
                        (<$bits>::from(Self::N != $crate::NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks,
                    );
                }

                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Self::from_bits(magnitude.min(i64::from(Self::HUGE.to_bits())) as $bits | sign_bit)
            }

            /// Fast conversion to [`f32`]
            ///
            /// This method serves as a shortcut if conversion to [`f32`] is
            /// lossless.
            fn fast_to_f32(self) -> f32 {
                let sign = if self.is_sign_negative() { -1.0 } else { 1.0 };
                let magnitude = self.to_bits() & Self::ABS_MASK;

                if self.is_nan() {
                    return f32::NAN.copysign(sign);
                }
                if self.is_infinite() {
                    return f32::INFINITY * sign;
                }
                if magnitude < 1 << Self::M {
                    #[allow(clippy::cast_possible_wrap)]
                    let shift = Self::MIN_EXP - Self::MANTISSA_DIGITS as i32;
                    #[allow(clippy::cast_possible_truncation)]
                    return ($crate::detail::exp2i(shift) * f64::from(sign) * f64::from(magnitude)) as f32;
                }
                let shift = f32::MANTISSA_DIGITS - Self::MANTISSA_DIGITS;
                #[allow(clippy::cast_sign_loss)]
                let diff = (Self::MIN_EXP - f32::MIN_EXP) as u32;
                let diff = diff << (f32::MANTISSA_DIGITS - 1);
                let sign = u32::from(self.is_sign_negative()) << 31;
                f32::from_bits(((u32::from(magnitude) << shift) + diff) | sign)
            }

            /// Fast conversion to [`f64`]
            ///
            /// This method serves as a shortcut if conversion to [`f64`] is
            /// lossless.
            fn fast_to_f64(self) -> f64 {
                let sign = if self.is_sign_negative() { -1.0 } else { 1.0 };
                let magnitude = self.to_bits() & Self::ABS_MASK;

                if self.is_nan() {
                    return f64::NAN.copysign(sign);
                }
                if self.is_infinite() {
                    return f64::INFINITY * sign;
                }
                if magnitude < 1 << Self::M {
                    #[allow(clippy::cast_possible_wrap)]
                    let shift = Self::MIN_EXP - Self::MANTISSA_DIGITS as i32;
                    return $crate::detail::exp2i(shift) * sign * f64::from(magnitude);
                }
                let shift = f64::MANTISSA_DIGITS - Self::MANTISSA_DIGITS;
                #[allow(clippy::cast_sign_loss)]
                let diff = (Self::MIN_EXP - f64::MIN_EXP) as u64;
                let diff = diff << (f64::MANTISSA_DIGITS - 1);
                let sign = u64::from(self.is_sign_negative()) << 63;
                f64::from_bits(((u64::from(magnitude) << shift) + diff) | sign)
            }

            /// Lossy conversion to [`f64`]
            ///
            /// This variant assumes that the conversion is lossy only when the exponent
            /// is out of range.
            fn as_f64(self) -> f64 {
                let bias = (1 << (Self::E - 1)) - 1;
                let sign = if self.is_sign_negative() { -1.0 } else { 1.0 };
                let magnitude = self.abs().to_bits();

                if self.is_nan() {
                    return f64::NAN.copysign(sign);
                }
                if self.is_infinite() {
                    return f64::INFINITY * sign;
                }
                if i32::from(magnitude) >= (f64::MAX_EXP + bias) << Self::M {
                    return f64::INFINITY * sign;
                }
                if magnitude < 1 << Self::M {
                    #[allow(clippy::cast_possible_wrap)]
                    let shift = Self::MIN_EXP - Self::MANTISSA_DIGITS as i32;
                    return $crate::detail::exp2i(shift) * sign * f64::from(magnitude);
                }
                if i32::from(magnitude >> Self::M) < f64::MIN_EXP + bias {
                    let significand = (magnitude & ((1 << Self::M) - 1)) | 1 << Self::M;
                    let exponent = i32::from(magnitude >> Self::M) - bias;
                    #[allow(clippy::cast_possible_wrap)]
                    return $crate::detail::exp2i(exponent - Self::M as i32) * sign * f64::from(significand);
                }
                let shift = f64::MANTISSA_DIGITS - Self::MANTISSA_DIGITS;
                #[allow(clippy::cast_sign_loss)]
                let diff = (Self::MIN_EXP - f64::MIN_EXP) as u64;
                let diff = diff << (f64::MANTISSA_DIGITS - 1);
                let sign = u64::from(self.is_sign_negative()) << 63;
                f64::from_bits(((u64::from(magnitude) << shift) + diff) | sign)
            }

            /// Best effort conversion to [`f64`]
            #[must_use]
            pub fn to_f64(self) -> f64 {
                let lossless = f64::MANTISSA_DIGITS >= Self::MANTISSA_DIGITS
                    && f64::MAX_EXP >= Self::MAX_EXP
                    && f64::MIN_EXP <= Self::MIN_EXP;

                if lossless {
                    self.fast_to_f64()
                } else {
                    self.as_f64()
                }
            }

            /// Best effort conversion to [`f32`]
            #[must_use]
            pub fn to_f32(self) -> f32 {
                let lossless = f32::MANTISSA_DIGITS >= Self::MANTISSA_DIGITS
                    && f32::MAX_EXP >= Self::MAX_EXP
                    && f32::MIN_EXP <= Self::MIN_EXP;

                if lossless {
                    return self.fast_to_f32();
                }
                // Conversion to `f64` is lossy only when then exponent width is
                // too large.  In this case, a second conversion to `f32` is
                // safe.
                #[allow(clippy::cast_possible_truncation)]
                return self.to_f64() as f32;
            }
        }

        const _: () = assert!($name::BITWIDTH <= 16);
        const _: () = assert!($name::E >= 2);
        const _: () = assert!($name::M > 0 || !matches!($name::N, $crate::NanStyle::IEEE));
        const _: () = assert!($name::MAX_EXP >= 1);
        const _: () = assert!($name::MIN_EXP <= 1);

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                let eq = self.0 == other.0 && !self.is_nan();
                eq || !matches!(Self::N, $crate::NanStyle::FNUZ) && (self.0 | other.0) & Self::ABS_MASK == 0
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                if self.is_nan() || other.is_nan() {
                    return None;
                }
                if self == other {
                    return Some(core::cmp::Ordering::Equal);
                }

                let sign = (self.0 | other.0) >> (Self::E + Self::M) & 1 == 1;

                Some(if (self.0 > other.0) ^ sign {
                    core::cmp::Ordering::Greater
                } else {
                    core::cmp::Ordering::Less
                })
            }
        }

        impl core::ops::Neg for $name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                let flag = matches!(Self::N, $crate::NanStyle::FNUZ) && self.0 & Self::ABS_MASK == 0;
                let switch = <$bits>::from(!flag) << (Self::E + Self::M);
                Self(self.0 ^ switch)
            }
        }

        impl $crate::Minifloat for $name {
            type Bits = $bits;
            const E: u32 = $e;
            const M: u32 = $m;
            const B: i32 = $b;
            const N: $crate::NanStyle = $crate::NanStyle::$n;

            const NAN: Self = Self::NAN;
            const HUGE: Self = Self::HUGE;
            const MAX: Self = Self::MAX;
            const TINY: Self = Self::TINY;
            const MIN_POSITIVE: Self = Self::MIN_POSITIVE;
            const EPSILON: Self = Self::EPSILON;
            const MIN: Self = Self::MIN;

            fn from_bits(v: Self::Bits) -> Self {
                Self::from_bits(v)
            }

            fn to_bits(self) -> Self::Bits {
                self.to_bits()
            }

            fn total_cmp(&self, other: &Self) -> core::cmp::Ordering {
                Self::total_cmp_key(self.0).cmp(&Self::total_cmp_key(other.0))
            }

            fn is_nan(self) -> bool {
                self.is_nan()
            }

            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }

            fn classify(self) -> core::num::FpCategory {
                self.classify()
            }

            fn abs(self) -> Self {
                self.abs()
            }

            fn is_sign_positive(self) -> bool {
                self.is_sign_positive()
            }

            fn is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
        }

        $crate::__conditionally_define_infinities!(impl $name, $n);
    };
    ($vis:vis struct $name:ident($bits:tt): $e:expr, $m:expr, $n:ident) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, (1 << ($e - 1)) - 1, $n);
    };
    ($vis:vis struct $name:ident($bits:tt): $e:expr, $m:expr, $b:expr) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, $b, IEEE);
    };
    ($vis:vis struct $name:ident($bits:tt): $e:expr, $m:expr) => {
        $crate::minifloat!($vis struct $name($bits): $e, $m, (1 << ($e - 1)) - 1, IEEE);
    };
}

minifloat!(pub struct F16(u16): 5, 10);
minifloat!(pub struct BF16(u16): 8, 7);
