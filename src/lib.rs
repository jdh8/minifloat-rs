// This file is part of the minifloat project.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! A const generic library for minifloats
//!
//! This crate provides emulation of minifloats up to 16 bits.  This is done
//! with two generic structs, [`F8`] and [`F16`], which take up to 8 and 16
//! bits of storage respectively.  Many parameters are configurable, including
//!
//! - Exponent width
//! - Significand (mantissa) precision
//! - ([`F8`]-only) Exponent bias
//! - ([`F8`]-only) NaN encodings: [IEEE][NanStyle::IEEE], [FN][NanStyle::FN], or [FNUZ][NanStyle::FNUZ]
//!
//! Note that there is always a sign bit, so [`F8<4, 3>`] already uses up all
//! 8 bits: 1 sign bit, 4 exponent bits, and 3 significand bits.

#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![warn(missing_docs)]

mod test;
use core::cmp::Ordering;
use core::marker::ConstParamTy;
use core::mem;
use core::num::FpCategory;
use core::ops::Neg;

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
    /// zero.  There are no infinities.  The negative zero (&minus;0)
    /// representation is reserved for NaN.  As a result, there is only one
    /// (+0) unsigned zero.
    FNUZ,
}

/// Minifloat taking up to 8 bits with configurable bias and NaN encoding
///
/// * `E`: exponent width
/// * `M`: significand (mantissa) precision
/// * `N`: NaN encoding style
/// * `B`: exponent bias, which defaults to 2<sup>`E`&minus;1</sup> &minus; 1
#[derive(Debug, Clone, Copy)]
pub struct F8<
    const E: u32,
    const M: u32,
    const N: NanStyle = { NanStyle::IEEE },
    const B: i32 = { (1 << (E - 1)) - 1 },
>(u8);

/// Minifloat taking up to 16 bits
///
/// * `E`: exponent width
/// * `M`: significand (mantissa) precision
#[derive(Debug, Clone, Copy)]
pub struct F16<const E: u32, const M: u32>(u16);

/// [`F16<5, 10>`], IEEE binary16, half precision
#[allow(non_camel_case_types)]
pub type f16 = F16<5, 10>;

/// [`F16<8, 7>`], bfloat16 format
#[allow(non_camel_case_types)]
pub type bf16 = F16<8, 7>;

/// Fast 2<sup>`x`</sup> with bit manipulation
fn fast_exp2(x: i32) -> f64 {
    f64::from_bits(match 0x3FF + x {
        0x800.. => 0x7FF << 52,
        #[allow(clippy::cast_sign_loss)]
        s @ 1..=0x7FF => (s as u64) << 52,
        s @ -51..=0 => 1 << (51 + s),
        _ => 0,
    })
}

macro_rules! define_round_to_precision {
    ($name:ident, $f:ty) => {
        /// Round `x` to the nearest representable value with `M` bits of precision
        fn $name<const M: u32>(x: $f) -> $f {
            let x = x.to_bits();
            let shift = <$f>::MANTISSA_DIGITS - 1 - M;
            let ulp = 1 << shift;
            let bias = (ulp >> 1) - (!(x >> shift) & 1);
            <$f>::from_bits((x + bias) & !(ulp - 1))
        }
    };
}

define_round_to_precision!(round_f32_to_precision, f32);
define_round_to_precision!(round_f64_to_precision, f64);

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> F8<E, M, N, B> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 8);
    const _HAS_EXPONENT: () = assert!(E > 0);

    /// The radix of the internal representation
    pub const RADIX: u32 = 2;

    /// The number of digits in the significand, including the implicit leading bit
    pub const MANTISSA_DIGITS: u32 = M + 1;

    /// The maximum exponent
    ///
    /// Normal numbers < 1 &times; 2<sup>`MAX_EXP`</sup>.
    pub const MAX_EXP: i32 = (1 << E) - B - matches!(N, NanStyle::IEEE) as i32;

    /// One greater than the minimum normal exponent
    ///
    /// Normal numbers ≥ 0.5 &times; 2<sup>`MIN_EXP`</sup>.
    ///
    /// This quirk comes from C macros `FLT_MIN_EXP` and friends.  However, it
    /// is no big deal to mistake it since [[`MIN_POSITIVE`][Self::MIN_POSITIVE],
    /// 2 &times; `MIN_POSITIVE`] is a buffer zone where numbers can be
    /// interpreted as normal or subnormal.
    pub const MIN_EXP: i32 = 2 - B;

    const _IS_LOSSLESS_INTO_F32: () =
        assert!(Self::MAX_EXP <= f32::MAX_EXP && Self::MIN_EXP >= f32::MIN_EXP);

    /// One representation of NaN
    pub const NAN: Self = Self(match N {
        NanStyle::IEEE => ((1 << (E + 1)) - 1) << (M - 1),
        NanStyle::FN => (1 << (E + M)) - 1,
        NanStyle::FNUZ => 1 << (E + M),
    });

    /// The largest number of this type
    ///
    /// This value would be +∞ if the type has infinities.  Otherwise, it is
    /// the maximum finite representation.  This value is also the result of
    /// a positive overflow.
    pub const HUGE: Self = Self(match N {
        NanStyle::IEEE => ((1 << E) - 1) << M,
        NanStyle::FN => (1 << (E + M)) - 2,
        NanStyle::FNUZ => (1 << (E + M)) - 1,
    });

    /// The maximum finite number
    pub const MAX: Self = Self(Self::HUGE.0 - matches!(N, NanStyle::IEEE) as u8);

    /// The smallest positive (subnormal) number
    pub const TINY: Self = Self(1);

    /// The smallest positive normal number
    ///
    /// Equal to 2<sup>[`MIN_EXP`][Self::MIN_EXP]&minus;1</sup>
    pub const MIN_POSITIVE: Self = Self(1 << M);

    /// The minimum finite number
    ///
    /// Equal to &minus;[`MAX`][Self::MAX]
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    const ABS_MASK: u8 = (1 << (E + M)) - 1;

    /// Raw transmutation from `u8`
    #[must_use]
    pub const fn from_bits(v: u8) -> Self {
        let mask = if E + M >= 7 {
            0xFF
        } else {
            (1 << (E + M + 1)) - 1
        };
        Self(v & mask)
    }

    /// Raw transmutation to `u8`
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Check if the value is NaN
    #[must_use]
    pub const fn is_nan(self) -> bool {
        match N {
            NanStyle::IEEE => self.0 & Self::ABS_MASK > Self::HUGE.0,
            NanStyle::FN => self.0 & Self::ABS_MASK == Self::NAN.0,
            NanStyle::FNUZ => self.0 == Self::NAN.0,
        }
    }

    /// Check if the value is positive or negative infinity
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        matches!(N, NanStyle::IEEE) && self.0 & Self::ABS_MASK == Self::HUGE.0
    }

    /// Check if the value is finite, i.e. neither infinite nor NaN
    #[must_use]
    pub const fn is_finite(self) -> bool {
        match N {
            NanStyle::IEEE => self.0 & Self::ABS_MASK < Self::HUGE.0,
            _ => !self.is_nan(),
        }
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
        if self.is_nan() {
            FpCategory::Nan
        } else if self.is_infinite() {
            FpCategory::Infinite
        } else {
            let exp_mask = ((1 << E) - 1) << M;
            let man_mask = (1 << M) - 1;

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
}

impl<const E: u32, const M: u32, const B: i32> F8<E, M, { NanStyle::IEEE }, B> {
    /// Positive infinity (+∞)
    pub const INFINITY: Self = Self(((1 << E) - 1) << M);

    /// Negative infinity (&minus;∞)
    pub const NEG_INFINITY: Self = Self(Self::INFINITY.0 | 1 << (E + M));
}

impl<const E: u32, const M: u32> F16<E, M> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 16);
    const _HAS_EXPONENT: () = assert!(E > 0);

    /// The radix of the internal representation
    pub const RADIX: u32 = 2;

    /// The number of digits in the significand, including the implicit leading bit
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
    /// Equal to (1 &minus; 2<sup>&minus;[`MANTISSA_DIGITS`][Self::MANTISSA_DIGITS]</sup>) 2<sup>[`MAX_EXP`][Self::MAX_EXP]</sup>
    pub const MAX: Self = Self(Self::INFINITY.0 - 1);

    /// The smallest positive (subnormal) number
    pub const TINY: Self = Self(1);

    /// The smallest positive normal number
    ///
    /// Equal to 2<sup>[`MIN_EXP`][Self::MIN_EXP]&minus;1</sup>
    pub const MIN_POSITIVE: Self = Self(1 << M);

    /// The minimum finite number
    ///
    /// Equal to &minus;[`MAX`][Self::MAX]
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    const _IS_LOSSLESS_INTO_F32: () =
        assert!(Self::MAX_EXP <= f32::MAX_EXP && Self::MIN_EXP >= f32::MIN_EXP);

    const ABS_MASK: u16 = (1 << (E + M)) - 1;

    /// Raw transmutation from `u16`
    #[must_use]
    pub const fn from_bits(v: u16) -> Self {
        let mask = if E + M >= 15 {
            0xFFFF
        } else {
            (1 << (E + M + 1)) - 1
        };
        Self(v & mask)
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
}

/// Mutual transmutation
///
/// This trait provides an interface of mutual raw transmutation.  The methods
/// default to using [`core::mem::transmute_copy`] for the conversion, but you
/// can override them for safer implementations.
///
/// In this crate, all [`F8`] types implement `Transmute<u8>`, and all [`F16`]
/// types implement `Transmute<u16>`.
pub trait Transmute<T>: Copy {
    /// Assert the same size between `T` and `Self`
    ///
    /// Do not override this constant.
    const _SAME_SIZE: () = assert!(mem::size_of::<T>() == mem::size_of::<Self>());

    /// Raw transmutation from `T`
    fn from_bits(v: T) -> Self {
        unsafe { mem::transmute_copy(&v) }
    }

    /// Raw transmutation to `T`
    fn to_bits(self) -> T {
        unsafe { mem::transmute_copy(&self) }
    }
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> Transmute<u8> for F8<E, M, N, B> {
    fn from_bits(v: u8) -> Self {
        Self::from_bits(v)
    }

    fn to_bits(self) -> u8 {
        self.0
    }
}

impl<const E: u32, const M: u32> Transmute<u16> for F16<E, M> {
    fn from_bits(v: u16) -> Self {
        Self::from_bits(v)
    }

    fn to_bits(self) -> u16 {
        self.0
    }
}

macro_rules! define_into_f32 {
    ($name:ident, $f:ty) => {
        fn $name(x: $f) -> f32 {
            let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
            let magnitude = x.0 & <$f>::ABS_MASK;

            if x.is_nan() {
                return f32::NAN * sign;
            }
            if x.is_infinite() {
                return f32::INFINITY * sign;
            }
            if magnitude < 1 << M {
                #[allow(clippy::cast_possible_wrap)]
                let shift = <$f>::MIN_EXP - <$f>::MANTISSA_DIGITS as i32;
                #[allow(clippy::cast_possible_truncation)]
                return (fast_exp2(shift) * f64::from(sign) * f64::from(magnitude)) as f32;
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

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> From<F8<E, M, N, B>> for f32 {
    define_into_f32!(from, F8<E, M, N, B>);
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> From<F8<E, M, N, B>> for f64 {
    fn from(x: F8<E, M, N, B>) -> Self {
        f32::from(x).into()
    }
}

impl<const E: u32, const M: u32> From<F16<E, M>> for f32 {
    define_into_f32!(from, F16<E, M>);
}

impl<const E: u32, const M: u32> From<F16<E, M>> for f64 {
    fn from(x: F16<E, M>) -> Self {
        f32::from(x).into()
    }
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> PartialEq for F8<E, M, N, B> {
    fn eq(&self, other: &Self) -> bool {
        let eq = self.0 == other.0 && !self.is_nan();
        eq || !matches!(N, NanStyle::FNUZ) && (self.0 | other.0) & Self::ABS_MASK == 0
    }
}

impl<const E: u32, const M: u32> PartialEq for F16<E, M> {
    fn eq(&self, other: &Self) -> bool {
        let eq = self.0 == other.0 && !self.is_nan();
        eq || (self.0 | other.0) & Self::ABS_MASK == 0
    }
}

macro_rules! impl_partial_cmp {
    () => {
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
    };
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> PartialOrd for F8<E, M, N, B> {
    impl_partial_cmp!();
}

impl<const E: u32, const M: u32> PartialOrd for F16<E, M> {
    impl_partial_cmp!();
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> Neg for F8<E, M, N, B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let flag = matches!(N, NanStyle::FNUZ) && self.0 & Self::ABS_MASK == 0;
        let switch = u8::from(!flag) << (E + M);
        Self(self.0 ^ switch)
    }
}

impl<const E: u32, const M: u32> Neg for F16<E, M> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0 ^ 1 << (E + M))
    }
}

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

    /// The largest number of this type
    ///
    /// This value would be +∞ if the type has infinities.  Otherwise, it is
    /// the maximum finite representation.  This value is also the result of
    /// a positive overflow.
    const HUGE: Self;

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
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> Minifloat for F8<E, M, N, B> {
    const E: u32 = E;
    const M: u32 = M;
    const N: NanStyle = N;
    const B: i32 = B;

    const HUGE: Self = Self::HUGE;

    #[allow(clippy::cast_possible_wrap)]
    fn from_f32(x: f32) -> Self {
        let bits = round_f32_to_precision::<M>(x).to_bits();
        let sign_bit = ((bits >> 31) as u8) << (E + M);

        if x.is_nan() {
            return Self(Self::NAN.0 | sign_bit);
        }

        let diff = (Self::MIN_EXP - f32::MIN_EXP) << M;
        let magnitude = bits << 1 >> (f32::MANTISSA_DIGITS - M);
        let magnitude = magnitude as i32 - diff;

        if magnitude < 1 << M {
            let ticks =
                f64::from(x.abs()) * fast_exp2(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ticks = ticks.round_ties_even() as u8;
            return Self((u8::from(N != NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self(magnitude.min(i32::from(Self::HUGE.0)) as u8 | sign_bit)
    }

    #[allow(clippy::cast_possible_wrap)]
    fn from_f64(x: f64) -> Self {
        let bits = round_f64_to_precision::<M>(x).to_bits();
        let sign_bit = ((bits >> 63) as u8) << (E + M);

        if x.is_nan() {
            return Self(Self::NAN.0 | sign_bit);
        }

        let diff = i64::from(Self::MIN_EXP - f64::MIN_EXP) << M;
        let magnitude = bits << 1 >> (f64::MANTISSA_DIGITS - M);
        let magnitude = magnitude as i64 - diff;

        if magnitude < 1 << M {
            let ticks = x.abs() * fast_exp2(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ticks = ticks.round_ties_even() as u8;
            return Self((u8::from(N != NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self(magnitude.min(i64::from(Self::HUGE.0)) as u8 | sign_bit)
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
}

impl<const E: u32, const M: u32> Minifloat for F16<E, M> {
    const E: u32 = E;
    const M: u32 = M;

    const HUGE: Self = Self::HUGE;

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
            let ticks =
                f64::from(x.abs()) * fast_exp2(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
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
            let ticks = x.abs() * fast_exp2(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
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
}
