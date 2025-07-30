// This file is part of the minifloat project.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod example;
mod most8;

use core::f64::consts::LOG10_2;
pub use most8::Most8;

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

/// Fast 2<sup>`x`</sup> with bit manipulation
const fn exp2i(x: i32) -> f64 {
    f64::from_bits(match 0x3FF + x {
        0x800.. => 0x7FF << 52,
        #[allow(clippy::cast_sign_loss)]
        s @ 1..=0x7FF => (s as u64) << 52,
        s @ -51..=0 => 1 << (51 + s),
        _ => 0,
    })
}

/// Round `x` to the nearest representable value with `M` bits of precision
const fn round_f32_to_precision<const M: u32>(x: f32) -> f32 {
    let x = x.to_bits();
    let shift = f32::MANTISSA_DIGITS - 1 - M;
    let ulp = 1 << shift;
    let bias = (ulp >> 1) - (!(x >> shift) & 1);
    f32::from_bits((x + bias) & !(ulp - 1))
}

/// Round `x` to the nearest representable value with `M` bits of precision
const fn round_f64_to_precision<const M: u32>(x: f64) -> f64 {
    let x = x.to_bits();
    let shift = f64::MANTISSA_DIGITS - 1 - M;
    let ulp = 1 << shift;
    let bias = (ulp >> 1) - (!(x >> shift) & 1);
    f64::from_bits((x + bias) & !(ulp - 1))
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

macro_rules! define_f64_from {
    ($name:ident, $f:ty) => {
        fn $name(x: $f) -> f64 {
            let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
            let magnitude = x.0 & <$f>::ABS_MASK;

            if x.is_nan() {
                return f64::NAN.copysign(sign);
            }
            if x.is_infinite() {
                return f64::INFINITY * sign;
            }
            if magnitude < 1 << M {
                #[allow(clippy::cast_possible_wrap)]
                let shift = <$f>::MIN_EXP - <$f>::MANTISSA_DIGITS as i32;
                return exp2i(shift) * sign * f64::from(magnitude);
            }
            let shift = f64::MANTISSA_DIGITS - <$f>::MANTISSA_DIGITS;
            #[allow(clippy::cast_sign_loss)]
            let diff = (<$f>::MIN_EXP - f64::MIN_EXP) as u64;
            let diff = diff << (f64::MANTISSA_DIGITS - 1);
            let sign = u64::from(x.is_sign_negative()) << 63;
            f64::from_bits(((u64::from(magnitude) << shift) + diff) | sign)
        }
    };
}

/*
fn as_f64<const E: u32, const M: u32>(x: Most16<E, M>) -> f64 {
    let bias = (1 << (E - 1)) - 1;
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let magnitude = x.abs().to_bits();

    if x.is_nan() {
        return f64::NAN.copysign(sign);
    }
    if x.is_infinite() {
        return f64::INFINITY * sign;
    }
    if i32::from(magnitude) >= (f64::MAX_EXP + bias) << M {
        return f64::INFINITY * sign;
    }
    if magnitude < 1 << M {
        #[allow(clippy::cast_possible_wrap)]
        let shift = Most16::<E, M>::MIN_EXP - Most16::<E, M>::MANTISSA_DIGITS as i32;
        return exp2i(shift) * sign * f64::from(magnitude);
    }
    if i32::from(magnitude >> M) < f64::MIN_EXP + bias {
        let significand = (magnitude & ((1 << M) - 1)) | 1 << M;
        let exponent = i32::from(magnitude >> M) - bias;
        #[allow(clippy::cast_possible_wrap)]
        return exp2i(exponent - M as i32) * sign * f64::from(significand);
    }
    let shift = f64::MANTISSA_DIGITS - Most16::<E, M>::MANTISSA_DIGITS;
    #[allow(clippy::cast_sign_loss)]
    let diff = (Most16::<E, M>::MIN_EXP - f64::MIN_EXP) as u64;
    let diff = diff << (f64::MANTISSA_DIGITS - 1);
    let sign = u64::from(x.is_sign_negative()) << 63;
    f64::from_bits(((u64::from(magnitude) << shift) + diff) | sign)
} // */

/// Internal macro to select the correct sized trait implementation
///
/// This macro needs to be public for [`minifloat!`] to invoke, but it is not
/// intended for general use.
#[doc(hidden)]
#[macro_export]
macro_rules! select_sized_trait {
    (u8, $name:ident, $e:expr, $m:expr) => {
        impl $crate::Most8<$e, $m> for $name {
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
    ($($t:tt)*) => {};
}

/// Define a minifloat taking up to 8 bits
///
/// * `$e`: exponent bit-width
/// * `$m`: explicit significand (mantissa) bit-width
/// * `$b`: exponent bias, which defaults to 2<sup>`$e`&minus;1</sup> &minus; 1
/// * `$n`: NaN encoding style
///
/// Constraints:
/// * `$e` + `$m` < 8 (there is always a sign bit)
/// * `$e` ≥ 2 (or use an integer type instead)
/// * `$m` > 0 if `$n` is [`IEEE`][NanStyle::IEEE] (∞ ≠ NaN)
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

        $crate::select_sized_trait!($bits, $name, $e, $m);
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

minifloat!(pub struct f16(u16): 5, 10);
minifloat!(pub struct bf16(u16): 8, 7);
