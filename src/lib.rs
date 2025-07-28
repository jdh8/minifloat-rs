// This file is part of the minifloat project.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![doc = include_str!("../README.md")]
#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features, private_bounds)]
#![warn(missing_docs)]

mod test;
use core::cmp::Ordering;
use core::f64::consts::LOG10_2;
use core::marker::ConstParamTy;
use core::mem;
use core::num::FpCategory;
use core::ops::Neg;
use num_traits::{AsPrimitive, ToPrimitive};

/// NaN encoding style
///
/// The variants follow [LLVM/MLIR naming conventions][llvm] derived from
/// their differences to [IEEE 754][ieee].
///
/// [llvm]: https://llvm.org/doxygen/structllvm_1_1APFloatBase.html
/// [ieee]: https://en.wikipedia.org/wiki/IEEE_754
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, ConstParamTy, PartialEq, Eq)]
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

/// Minifloat taking up to 16 bits
///
/// * `E`: exponent width
/// * `M`: significand (mantissa) precision
///
/// Constraints:
/// * `E` + `M` < 16 (there is always a sign bit)
/// * `E` + `M` ≥ 8 (otherwise use [`F8`] instead)
/// * `E` ≥ 2 (or use an integer type instead)
/// * `M` > 0 (∞ ≠ NaN)
/// * 1.0 is normal
#[derive(Debug, Clone, Copy, Default)]
pub struct F16<const E: u32, const M: u32>(u16);

/// [`F16<5, 10>`], IEEE binary16, half precision
#[allow(non_camel_case_types)]
pub type f16 = F16<5, 10>;

/// [`F16<8, 7>`], bfloat16 format
#[allow(non_camel_case_types)]
pub type bf16 = F16<8, 7>;

/// Check a condition at compile time
struct Check<const COND: bool>;

/// The trait for [`Check<true>`]
trait True {}

impl True for Check<true> {}

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

impl<const E: u32, const M: u32> F16<E, M> {
    /// Check if the parameters are valid
    const VALID: bool = E + M < 16 && E + M >= 8 && E >= 2 && M > 0;

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
    pub const fn from_bits(v: u16) -> Self
    where
        Check<{ Self::VALID }>: True,
    {
        Self(0xFFFF >> (15 - E - M) & v)
    }

    /// Raw transmutation to `u16`
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Check if the value is NaN
    #[must_use]
    pub const fn is_nan(self) -> bool
    where
        Check<{ Self::VALID }>: True,
    {
        self.0 & Self::ABS_MASK > Self::INFINITY.0
    }

    /// Check if the value is positive or negative infinity
    #[must_use]
    pub const fn is_infinite(self) -> bool
    where
        Check<{ Self::VALID }>: True,
    {
        self.0 & Self::ABS_MASK == Self::INFINITY.0
    }

    /// Check if the value is finite, i.e. neither infinite nor NaN
    #[must_use]
    pub const fn is_finite(self) -> bool
    where
        Check<{ Self::VALID }>: True,
    {
        self.0 & Self::ABS_MASK < Self::INFINITY.0
    }

    /// Check if the value is [subnormal]
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    pub const fn is_subnormal(self) -> bool
    where
        Check<{ Self::VALID }>: True,
    {
        matches!(self.classify(), FpCategory::Subnormal)
    }

    /// Check if the value is normal, i.e. not zero, [subnormal], infinite, or NaN
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    pub const fn is_normal(self) -> bool
    where
        Check<{ Self::VALID }>: True,
    {
        matches!(self.classify(), FpCategory::Normal)
    }

    /// Classify the value into a floating-point category
    ///
    /// If only one property is going to be tested, it is generally faster to
    /// use the specific predicate instead.
    #[must_use]
    pub const fn classify(self) -> FpCategory
    where
        Check<{ Self::VALID }>: True,
    {
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
}

trait SameSized<T> {}

impl<T, U> SameSized<T> for U where Check<{ mem::size_of::<T>() == mem::size_of::<Self>() }>: True {}

/// Mutual transmutation
///
/// This trait provides an interface of mutual raw transmutation.  The methods
/// default to using [`core::mem::transmute_copy`] for the conversion, but you
/// can override them for safer implementations.
///
/// In this crate, all [`F8`] types implement `Transmute<u8>`, and all [`F16`]
/// types implement `Transmute<u16>`.
pub trait Transmute<T>: Copy + SameSized<T> {
    /// Raw transmutation from `T`
    fn from_bits(v: T) -> Self {
        unsafe { mem::transmute_copy(&v) }
    }

    /// Raw transmutation to `T`
    fn to_bits(self) -> T {
        unsafe { mem::transmute_copy(&self) }
    }
}

impl<const E: u32, const M: u32> Transmute<u16> for F16<E, M>
where
    Check<{ Self::VALID }>: True,
    Self: SameSized<u16>,
{
    fn from_bits(v: u16) -> Self {
        Self::from_bits(v)
    }

    fn to_bits(self) -> u16 {
        self.0
    }
}

impl<const E: u32, const M: u32> PartialEq for F16<E, M>
where
    Check<{ Self::VALID }>: True,
{
    fn eq(&self, other: &Self) -> bool {
        let eq = self.0 == other.0 && !self.is_nan();
        eq || (self.0 | other.0) & Self::ABS_MASK == 0
    }
}

impl<const E: u32, const M: u32> PartialOrd for F16<E, M>
where
    Check<{ Self::VALID }>: True,
{
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

impl<const E: u32, const M: u32> Neg for F16<E, M> {
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

/// Generic trait for minifloat types
///
/// We are **not** going to implement [`num_traits::Float`][flt] because:
///
/// 1. [`FN`][NanStyle::FN] and [`FNUZ`][NanStyle::FNUZ] types do not have infinities.
/// 2. [`FNUZ`][NanStyle::FNUZ] types do not have a negative zero.
/// 3. We don't have plans for [arithmetic operations][ops] yet.
///
/// [flt]: https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html
/// [ops]: https://docs.rs/num-traits/latest/num_traits/trait.NumOps.html
pub trait Minifloat: Copy + PartialEq + PartialOrd + Neg<Output = Self> {
    /// Exponent width
    const E: u32;

    /// Significand (mantissa) precision
    const M: u32;

    /// NaN encoding style
    const N: NanStyle = NanStyle::IEEE;

    /// Exponent bias, which defaults to 2<sup>`E`&minus;1</sup> &minus; 1
    const B: i32 = (1 << (Self::E - 1)) - 1;

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
    const DIGITS: u32 = (Self::M as f64 * LOG10_2) as u32;

    /// Maximum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to floor(log<sub>10</sub>([`MAX`][Self::MAX]))
    #[allow(clippy::cast_possible_truncation)]
    const MAX_10_EXP: i32 = {
        let exponent = (1 << Self::E) - Self::B - matches!(Self::N, NanStyle::IEEE) as i32;
        let precision = Self::M + !matches!(Self::N, NanStyle::FN) as u32;
        let log2_max = exponent as f64 + LOG2_SIGNIFICAND[precision as usize];
        (log2_max * LOG10_2) as i32
    };

    /// Minimum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to ceil(log<sub>10</sub>([`MIN_POSITIVE`][Self::MIN_POSITIVE]))
    #[allow(clippy::cast_possible_truncation)]
    const MIN_10_EXP: i32 = ((Self::MIN_EXP - 1) as f64 * LOG10_2) as i32;

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

    /// Check if the value is NaN
    #[must_use]
    fn is_nan(self) -> bool;

    /// Check if the value is positive or negative infinity
    #[must_use]
    fn is_infinite(self) -> bool;

    /// Check if the value is finite, i.e. neither infinite nor NaN
    #[must_use]
    fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Check if the value is [subnormal]
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    fn is_subnormal(self) -> bool {
        matches!(self.classify(), FpCategory::Subnormal)
    }

    /// Check if the value is normal, i.e. not zero, [subnormal], infinite, or NaN
    ///
    /// [subnormal]: https://en.wikipedia.org/wiki/Subnormal_number
    #[must_use]
    fn is_normal(self) -> bool {
        matches!(self.classify(), FpCategory::Normal)
    }

    /// Classify the value into a floating-point category
    ///
    /// If only one property is going to be tested, it is generally faster to
    /// use the specific predicate instead.
    #[must_use]
    fn classify(self) -> FpCategory;

    /// Check if the sign bit is clear
    #[must_use]
    fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Check if the sign bit is set
    #[must_use]
    fn is_sign_negative(self) -> bool;

    /// Get the maximum of two numbers, ignoring NaN
    #[must_use]
    fn max(self, other: Self) -> Self {
        if self >= other || other.is_nan() {
            self
        } else {
            other
        }
    }

    /// Get the minimum of two numbers, ignoring NaN
    #[must_use]
    fn min(self, other: Self) -> Self {
        if self <= other || other.is_nan() {
            self
        } else {
            other
        }
    }

    /// IEEE 754 total-ordering predicate
    ///
    /// The normative definition is lengthy, but it is essentially comparing
    /// sign-magnitude notations.
    ///
    /// See also [`f32::total_cmp`],
    /// <https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate>
    #[must_use]
    fn total_cmp(&self, other: &Self) -> Ordering;

    /// Get the maximum of two numbers, propagating NaN
    ///
    /// For this operation, -0.0 is considered to be less than +0.0 as
    /// specified in IEEE 754-2019.
    #[must_use]
    fn maximum(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::NAN
        } else {
            core::cmp::max_by(self, other, Self::total_cmp)
        }
    }

    /// Get the minimum of two numbers, propagating NaN
    ///
    /// For this operation, -0.0 is considered to be less than +0.0 as
    /// specified in IEEE 754-2019.
    #[must_use]
    fn minimum(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::NAN
        } else {
            core::cmp::min_by(self, other, Self::total_cmp)
        }
    }

    /// Compute the absolute value
    #[must_use]
    fn abs(self) -> Self {
        if self.is_sign_negative() {
            -self
        } else {
            self
        }
    }
}

impl<const E: u32, const M: u32> Minifloat for F16<E, M>
where
    Check<{ Self::VALID }>: True,
{
    const E: u32 = E;
    const M: u32 = M;

    const NAN: Self = Self::NAN;
    const HUGE: Self = Self::HUGE;
    const MAX: Self = Self::MAX;
    const TINY: Self = Self::TINY;
    const MIN_POSITIVE: Self = Self::MIN_POSITIVE;
    const EPSILON: Self = Self::EPSILON;
    const MIN: Self = Self::MIN;

    #[allow(clippy::cast_possible_wrap)]
    fn from_f32(x: f32) -> Self {
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

    #[allow(clippy::cast_possible_wrap)]
    fn from_f64(x: f64) -> Self {
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

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_sign_positive(self) -> bool {
        self.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.is_sign_negative()
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        Self::total_cmp_key(self.0).cmp(&Self::total_cmp_key(other.0))
    }

    fn abs(self) -> Self {
        self.abs()
    }
}

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

/// Lossless conversion to `f32`
///
/// Enabled only when every value of the source type is representable in `f32`.
impl<const E: u32, const M: u32> From<F16<E, M>> for f32
where
    Check<{ F16::<E, M>::VALID }>: True,
    Check<{ F16::<E, M>::MAX_EXP <= Self::MAX_EXP }>: True,
    Check<{ F16::<E, M>::MIN_EXP >= Self::MIN_EXP }>: True,
{
    define_f32_from!(from, F16<E, M>);
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

/// Lossless conversion to `f64`
///
/// Enabled only when every value of the source type is representable in `f64`.
impl<const E: u32, const M: u32> From<F16<E, M>> for f64
where
    Check<{ F16::<E, M>::VALID }>: True,
    Check<{ F16::<E, M>::MAX_EXP <= Self::MAX_EXP }>: True,
    Check<{ F16::<E, M>::MIN_EXP >= Self::MIN_EXP }>: True,
{
    define_f64_from!(from, F16<E, M>);
}

fn as_f64<T: Minifloat + Transmute<u16>>(x: T) -> f64 {
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let magnitude = x.abs().to_bits();

    if x.is_nan() {
        return f64::NAN.copysign(sign);
    }
    if x.is_infinite() {
        return f64::INFINITY * sign;
    }
    if i32::from(magnitude) >= (f64::MAX_EXP + T::B) << T::M {
        return f64::INFINITY * sign;
    }
    if magnitude < 1 << T::M {
        #[allow(clippy::cast_possible_wrap)]
        let shift = T::MIN_EXP - T::MANTISSA_DIGITS as i32;
        return exp2i(shift) * sign * f64::from(magnitude);
    }
    if i32::from(magnitude >> T::M) < f64::MIN_EXP + T::B {
        let significand = (magnitude & ((1 << T::M) - 1)) | 1 << T::M;
        let exponent = i32::from(magnitude >> T::M) - T::B;
        #[allow(clippy::cast_possible_wrap)]
        return exp2i(exponent - T::M as i32) * sign * f64::from(significand);
    }
    let shift = f64::MANTISSA_DIGITS - T::MANTISSA_DIGITS;
    #[allow(clippy::cast_sign_loss)]
    let diff = (T::MIN_EXP - f64::MIN_EXP) as u64;
    let diff = diff << (f64::MANTISSA_DIGITS - 1);
    let sign = u64::from(x.is_sign_negative()) << 63;
    f64::from_bits(((u64::from(magnitude) << shift) + diff) | sign)
}

/// Possibly lossy conversion (via [`f64`])
impl<T: 'static + Copy, const E: u32, const M: u32> AsPrimitive<T> for F16<E, M>
where
    Self: Minifloat + Transmute<u16>,
    f64: AsPrimitive<T>,
{
    fn as_(self) -> T {
        as_f64(self).as_()
    }
}

/// Possibly lossy conversion (via [`f64`])
impl<const E: u32, const M: u32> ToPrimitive for F16<E, M>
where
    Self: Minifloat + Transmute<u16>,
{
    fn to_i64(&self) -> Option<i64> {
        as_f64(*self).to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        as_f64(*self).to_u64()
    }

    fn to_i128(&self) -> Option<i128> {
        as_f64(*self).to_i128()
    }

    fn to_u128(&self) -> Option<u128> {
        as_f64(*self).to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        as_f64(*self).to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(as_f64(*self))
    }
}

struct ConstU32<const N: u32>;

/// IEEE-like minifloat taking at most 8 bits
///
/// Types generated by [`struct_most_8!`] implement this trait.
pub trait FloatMost8: Sized {
    /// Explicit significand (mantissa) bitwidth
    ///
    /// This constant is required for internal technical reasons.
    /// [`Self::from_f32`] and [`Self::from_f64`] require `M` to break the loop.
    const M: u32;

    /// Exponent bitwidth
    const E: u32 = Self::E;

    /// Exponent bias
    const B: i32 = Self::B;

    /// NaN encoding style
    const N: NanStyle = Self::N;

    /// Total bitwidth
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
    const DIGITS: u32 = (Self::M as f64 * LOG10_2) as u32;

    /// Maximum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to floor(log<sub>10</sub>([`MAX`][Self::MAX]))
    #[allow(clippy::cast_possible_truncation)]
    const MAX_10_EXP: i32 = {
        let exponent = (1 << Self::E) - Self::B - matches!(Self::N, NanStyle::IEEE) as i32;
        let precision = Self::M + !matches!(Self::N, NanStyle::FN) as u32;
        let log2_max = exponent as f64 + LOG2_SIGNIFICAND[precision as usize];
        (log2_max * LOG10_2) as i32
    };

    /// Minimum <var>x</var> such that 10<sup>`x`</sup> is normal
    ///
    /// Equal to ceil(log<sub>10</sub>([`MIN_POSITIVE`][Self::MIN_POSITIVE]))
    #[allow(clippy::cast_possible_truncation)]
    const MIN_10_EXP: i32 = ((Self::MIN_EXP - 1) as f64 * LOG10_2) as i32;

    /// One representation of NaN
    const NAN: Self = Self::NAN;

    /// The largest number of this type
    ///
    /// This value would be +∞ if the type has infinities.  Otherwise, it is
    /// the maximum finite representation.  This value is also the result of
    /// a positive overflow.
    const HUGE: Self = Self::HUGE;

    /// The maximum finite number
    const MAX: Self = Self::MAX;

    /// The smallest positive (subnormal) number
    const TINY: Self = Self::TINY;

    /// The smallest positive normal number
    ///
    /// Equal to 2<sup>[`MIN_EXP`][Self::MIN_EXP]&minus;1</sup>
    const MIN_POSITIVE: Self = Self::MIN_POSITIVE;

    /// [Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)
    ///
    /// The difference between 1.0 and the next larger representable number.
    ///
    /// Equal to 2<sup>&minus;`M`</sup>.
    const EPSILON: Self = Self::EPSILON;

    /// The minimum finite number
    ///
    /// Equal to &minus;[`MAX`][Self::MAX]
    const MIN: Self = Self::MIN;

    /// Raw transmutation from `u8`
    #[must_use]
    fn from_bits(v: u8) -> Self;

    /// Raw transmutation to `u8`
    #[must_use]
    fn to_bits(self) -> u8;

    /// Probably lossy conversion from [`f32`]
    ///
    /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
    /// Other values are rounded to the nearest representable value.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    fn from_f32(x: f32) -> Self
    where
        ConstU32<{ Self::M }>:,
    {
        if x.is_nan() {
            let sign_bit = u8::from(x.is_sign_negative()) << (Self::E + Self::M);
            return Self::from_bits(Self::NAN.to_bits() | sign_bit);
        }

        let bits = round_f32_to_precision::<{ Self::M }>(x).to_bits();
        let sign_bit = ((bits >> 31) as u8) << (Self::E + Self::M);
        let diff = (Self::MIN_EXP - f32::MIN_EXP) << Self::M;
        let magnitude = bits << 1 >> (f32::MANTISSA_DIGITS - Self::M);
        let magnitude = magnitude as i32 - diff;

        if magnitude < 1 << Self::M {
            let ticks = f64::from(x.abs()) * exp2i(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ticks = ticks.round_ties_even() as u8;
            return Self::from_bits(
                (u8::from(Self::N != NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks,
            );
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self::from_bits(magnitude.min(i32::from(Self::HUGE.to_bits())) as u8 | sign_bit)
    }

    /// Probably lossy conversion from [`f64`]
    ///
    /// NaNs are preserved.  Overflows result in ±[`HUGE`][Self::HUGE].
    /// Other values are rounded to the nearest representable value.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    fn from_f64(x: f64) -> Self
    where
        ConstU32<{ Self::M }>:,
    {
        if x.is_nan() {
            let sign_bit = u8::from(x.is_sign_negative()) << (Self::E + Self::M);
            return Self::from_bits(Self::NAN.to_bits() | sign_bit);
        }

        let bits = round_f64_to_precision::<{ Self::M }>(x).to_bits();
        let sign_bit = ((bits >> 63) as u8) << (Self::E + Self::M);
        let diff = i64::from(Self::MIN_EXP - f64::MIN_EXP) << Self::M;
        let magnitude = bits << 1 >> (f64::MANTISSA_DIGITS - Self::M);
        let magnitude = magnitude as i64 - diff;

        if magnitude < 1 << Self::M {
            let ticks = x.abs() * exp2i(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ticks = ticks.round_ties_even() as u8;
            return Self::from_bits(
                (u8::from(Self::N != NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks,
            );
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self::from_bits(magnitude.min(i64::from(Self::HUGE.to_bits())) as u8 | sign_bit)
    }

    /// IEEE 754 total-ordering predicate
    ///
    /// The normative definition is lengthy, but it is essentially comparing
    /// sign-magnitude notations.
    ///
    /// See also [`f32::total_cmp`],
    /// <https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate>
    #[must_use]
    fn total_cmp(&self, other: &Self) -> Ordering;
}

/// Publicly define a minifloat taking up to 8 bits
///
/// * `$e`: exponent width
/// * `$m`: significand (mantissa) precision
/// * `$n`: NaN encoding style
/// * `$b`: exponent bias, which defaults to 2<sup>`$e`&minus;1</sup> &minus; 1
///
/// Constraints:
/// * `$e` + `$m` < 8 (there is always a sign bit)
/// * `$e` ≥ 2 (or use an integer type instead)
/// * `$m` > 0 if `$n` is [`IEEE`][NanStyle::IEEE] (∞ ≠ NaN)
#[macro_export]
macro_rules! struct_most_8 {
    ($name:ident, $e:expr, $m:expr, $b:expr, $n:ident) => {
        #[allow(non_camel_case_types)]
        #[doc = concat!("A minifloat with bit-layout S1E", $e, "M", $m)]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name(u8);

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
            pub const MAX: Self = Self(Self::HUGE.0 - matches!(Self::N, $crate::NanStyle::IEEE) as u8);

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
                s @ 1.. => (s as u8) << Self::M,
                s => 1 << (Self::M as i32 - 1 + s),
            });

            /// The minimum finite number
            ///
            /// Equal to &minus;[`MAX`][Self::MAX].
            pub const MIN: Self = Self(Self::MAX.0 | 1 << (Self::E + Self::M));

            /// Magnitude mask for internal usage
            const ABS_MASK: u8 = (1 << (Self::E + Self::M)) - 1;

            /// Raw transmutation from `u8`
            #[must_use]
            pub const fn from_bits(v: u8) -> Self {
                Self(0xFF >> (7 - Self::E - Self::M) & v)
            }

            /// Raw transmutation to `u8`
            #[must_use]
            pub const fn to_bits(self) -> u8 {
                self.0
            }

            /// Check if the value is NaN
            #[must_use]
            pub const fn is_nan(self) -> bool {
                match Self::N {
                    $crate::NanStyle::IEEE => self.0 & Self::ABS_MASK > Self::HUGE.0,
                    $crate::NanStyle::FN => self.0 & Self::ABS_MASK == Self::NAN.0,
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
            const fn total_cmp_key(x: u8) -> u8 {
                let sign = 1 << (Self::E + Self::M);
                let mask = ((x & sign) >> (Self::E + Self::M)) * (sign - 1);
                x ^ (sign | mask)
            }
        }

        const _: () = assert!($name::BITWIDTH <= 8);
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
                let switch = u8::from(!flag) << (Self::E + Self::M);
                Self(self.0 ^ switch)
            }
        }

        impl $crate::FloatMost8 for $name {
            const M: u32 = Self::M;

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
    ($name:ident, $e:expr, $m:expr, $n:ident) => {
        $crate::struct_most_8!($name, $e, $m, (1 << ($e - 1)) - 1, $n);
    };
    ($name:ident, $e:expr, $m:expr, $b:expr) => {
        $crate::struct_most_8!($name, $e, $m, $b, IEEE);
    };
    ($name:ident, $e:expr, $m:expr) => {
        $crate::struct_most_8!($name, $e, $m, (1 << ($e - 1)) - 1, IEEE);
    };
}

struct_most_8!(E4M3FN, 4, 3, FN);
