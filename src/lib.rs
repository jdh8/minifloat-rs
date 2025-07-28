// This file is part of the minifloat project.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod most8;

use core::cmp::Ordering;
use core::f64::consts::LOG10_2;
use core::num::FpCategory;
use core::ops::Neg;
pub use most8::{Most8, NanStyle};
use num_traits::{AsPrimitive, ToPrimitive};

/// Minifloat taking up to 16 bits
///
/// * `E`: exponent width
/// * `M`: significand (mantissa) precision
///
/// Constraints:
/// * `E` + `M` < 16 (there is always a sign bit)
/// * `E` + `M` ≥ 8 (otherwise use [`Most8`] instead)
/// * `E` ≥ 2 (or use an integer type instead)
/// * `M` > 0 (∞ ≠ NaN)
/// * 1.0 is normal
#[derive(Debug, Clone, Copy, Default)]
pub struct Most16<const E: u32, const M: u32>(u16);

/// [`Most16<5, 10>`], IEEE binary16, half precision
#[allow(non_camel_case_types)]
pub type f16 = Most16<5, 10>;

/// [`Most16<8, 7>`], bfloat16 format
#[allow(non_camel_case_types)]
pub type bf16 = Most16<8, 7>;

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

impl<const E: u32, const M: u32> Most16<E, M> {
    /// The radix of the internal representation
    pub const RADIX: u32 = 2;

    /// The number of digits in the significand, including the implicit leading bit
    ///
    /// Equal to `M` + 1
    pub const MANTISSA_DIGITS: u32 = M + 1;

    /// The maximum exponent
    ///
    /// Normal numbers < 1 &times; 2<sup>`MAX_EXP`</sup>.
    pub const MAX_EXP: i32 = 1 << (E - 1);

    /// One greater than the minimum normal exponent
    ///
    /// Normal numbers ≥ 0.5 &times; 2<sup>`MIN_EXP`</sup>.
    ///
    /// This quirk comes from C macros `FLT_MIN_EXP` and friends.  However, it
    /// is no big deal to mistake it since [[`MIN_POSITIVE`][Self::MIN_POSITIVE],
    /// 2 &times; `MIN_POSITIVE`] is a buffer zone where numbers can be
    /// interpreted as normal or subnormal.
    pub const MIN_EXP: i32 = 3 - Self::MAX_EXP;

    /// Positive infinity (+∞)
    pub const INFINITY: Self = Self(((1 << E) - 1) << M);

    /// Negative infinity (&minus;∞)
    pub const NEG_INFINITY: Self = Self(Self::INFINITY.0 | 1 << (E + M));

    /// One representation of NaN
    pub const NAN: Self = Self(((1 << (E + 1)) - 1) << (M - 1));

    /// Positive infinity, the largest number of this type
    pub const HUGE: Self = Self::INFINITY;

    /// The maximum finite number
    ///
    /// Equal to (1 &minus; 2<sup>&minus;[`MANTISSA_DIGITS`][Self::MANTISSA_DIGITS]</sup>) 2<sup>[`MAX_EXP`][Self::MAX_EXP]</sup>.
    pub const MAX: Self = Self(Self::INFINITY.0 - 1);

    /// The smallest positive (subnormal) number
    pub const TINY: Self = Self(1);

    /// The smallest positive normal number
    ///
    /// Equal to 2<sup>[`MIN_EXP`][Self::MIN_EXP]&minus;1</sup>.
    pub const MIN_POSITIVE: Self = Self(1 << M);

    /// [Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)
    ///
    /// The difference between 1.0 and the next larger representable number.
    ///
    /// Equal to 2<sup>&minus;`M`</sup>.
    #[allow(clippy::cast_possible_wrap)]
    pub const EPSILON: Self = Self(match (1 << (E - 1)) - 1 - M as i32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        s @ 1.. => (s as u16) << M,
        s => 1 << (M as i32 - 1 + s),
    });

    /// The minimum finite number
    ///
    /// Equal to &minus;[`MAX`][Self::MAX]
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    /// Magnitude mask for internal usage
    const ABS_MASK: u16 = (1 << (E + M)) - 1;

    /// Raw transmutation from `u16`
    #[must_use]
    pub const fn from_bits(v: u16) -> Self {
        Self(0xFFFF >> (15 - E - M) & v)
    }

    /// Raw transmutation to `u16`
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Check if the value is NaN
    #[must_use]
    pub const fn is_nan(self) -> bool {
        self.0 & Self::ABS_MASK > Self::INFINITY.0
    }

    /// Check if the value is positive or negative infinity
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        self.0 & Self::ABS_MASK == Self::INFINITY.0
    }

    /// Check if the value is finite, i.e. neither infinite nor NaN
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.0 & Self::ABS_MASK < Self::INFINITY.0
    }

    /// Check if the value is [subnormal]
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    pub const fn is_subnormal(self) -> bool {
        matches!(self.classify(), FpCategory::Subnormal)
    }

    /// Check if the value is normal, i.e. not zero, [subnormal], infinite, or NaN
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    pub const fn is_normal(self) -> bool {
        matches!(self.classify(), FpCategory::Normal)
    }

    /// Classify the value into a floating-point category
    ///
    /// If only one property is going to be tested, it is generally faster to
    /// use the specific predicate instead.
    #[must_use]
    pub const fn classify(self) -> FpCategory {
        let exp_mask = ((1 << E) - 1) << M;
        let man_mask = (1 << M) - 1;

        if self.0 & exp_mask == exp_mask {
            if self.0 & man_mask == 0 {
                FpCategory::Infinite
            } else {
                FpCategory::Nan
            }
        } else {
            match (self.0 & exp_mask, self.0 & man_mask) {
                (0, 0) => FpCategory::Zero,
                (0, _) => FpCategory::Subnormal,
                (_, _) => FpCategory::Normal,
            }
        }
    }

    /// Check if the sign bit is clear
    #[must_use]
    pub const fn is_sign_positive(self) -> bool {
        self.0 >> (E + M) & 1 == 0
    }

    /// Check if the sign bit is set
    #[must_use]
    pub const fn is_sign_negative(self) -> bool {
        self.0 >> (E + M) & 1 == 1
    }

    /// Map sign-magnitude notations to plain unsigned integers
    ///
    /// This serves as a hook for the [`Minifloat`] trait.
    const fn total_cmp_key(x: u16) -> u16 {
        let sign = 1 << (E + M);
        let mask = ((x & sign) >> (E + M)) * (sign - 1);
        x ^ (sign | mask)
    }

    /// Compute the absolute value
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0 & Self::ABS_MASK)
    }

    /// Probably lossy conversion from [`f32`]
    ///
    /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
    /// Other values are rounded to the nearest representable value.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_f32(x: f32) -> Self {
        let bits = round_f32_to_precision::<M>(x).to_bits();
        let sign_bit = ((bits >> 31) as u16) << (E + M);

        if x.is_nan() {
            return Self(Self::NAN.0 | sign_bit);
        }

        let diff = (Self::MIN_EXP - f32::MIN_EXP) << M;
        let magnitude = bits << 1 >> (f32::MANTISSA_DIGITS - M);
        let magnitude = magnitude as i32 - diff;

        if magnitude < 1 << M {
            let ticks = f64::from(x.abs()) * exp2i(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            return Self(ticks.round_ties_even() as u16 | sign_bit);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self(magnitude.min(i32::from(Self::HUGE.0)) as u16 | sign_bit)
    }

    /// Probably lossy conversion from [`f64`]
    ///
    /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
    /// Other values are rounded to the nearest representable value.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn from_f64(x: f64) -> Self {
        let bits = round_f64_to_precision::<M>(x).to_bits();
        let sign_bit = ((bits >> 63) as u16) << (E + M);

        if x.is_nan() {
            return Self(Self::NAN.0 | sign_bit);
        }

        let diff = i64::from(Self::MIN_EXP - f64::MIN_EXP) << M;
        let magnitude = bits << 1 >> (f64::MANTISSA_DIGITS - M);
        let magnitude = magnitude as i64 - diff;

        if magnitude < 1 << M {
            let ticks = x.abs() * exp2i(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            return Self(ticks.round_ties_even() as u16 | sign_bit);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self(magnitude.min(i64::from(Self::HUGE.0)) as u16 | sign_bit)
    }

    /// IEEE 754 total-ordering predicate
    ///
    /// The normative definition is lengthy, but it is essentially comparing
    /// sign-magnitude notations.
    ///
    /// See also [`f32::total_cmp`],
    /// <https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate>
    #[must_use]
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        Self::total_cmp_key(self.0).cmp(&Self::total_cmp_key(other.0))
    }
}

impl<const E: u32, const M: u32> PartialEq for Most16<E, M> {
    fn eq(&self, other: &Self) -> bool {
        let eq = self.0 == other.0 && !self.is_nan();
        eq || (self.0 | other.0) & Self::ABS_MASK == 0
    }
}

impl<const E: u32, const M: u32> PartialOrd for Most16<E, M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        if self == other {
            return Some(Ordering::Equal);
        }

        let sign = (self.0 | other.0) >> (E + M) & 1 == 1;

        Some(if (self.0 > other.0) ^ sign {
            Ordering::Greater
        } else {
            Ordering::Less
        })
    }
}

impl<const E: u32, const M: u32> Neg for Most16<E, M> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0 ^ 1 << (E + M))
    }
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

macro_rules! define_f32_from {
    ($name:ident, $f:ty) => {
        fn $name(x: $f) -> f32 {
            let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
            let magnitude = x.0 & <$f>::ABS_MASK;

            if x.is_nan() {
                return f32::NAN.copysign(sign);
            }
            if x.is_infinite() {
                return f32::INFINITY * sign;
            }
            if magnitude < 1 << M {
                #[allow(clippy::cast_possible_wrap)]
                let shift = <$f>::MIN_EXP - <$f>::MANTISSA_DIGITS as i32;
                #[allow(clippy::cast_possible_truncation)]
                return (exp2i(shift) * f64::from(sign) * f64::from(magnitude)) as f32;
            }
            let shift = f32::MANTISSA_DIGITS - <$f>::MANTISSA_DIGITS;
            #[allow(clippy::cast_sign_loss)]
            let diff = (<$f>::MIN_EXP - f32::MIN_EXP) as u32;
            let diff = diff << (f32::MANTISSA_DIGITS - 1);
            let sign = u32::from(x.is_sign_negative()) << 31;
            f32::from_bits(((u32::from(magnitude) << shift) + diff) | sign)
        }
    };
}

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
}
