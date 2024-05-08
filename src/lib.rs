// This file is part of the minifloat project.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod test;
use core::marker::ConstParamTy;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, ConstParamTy, PartialEq, Eq)]
pub enum NanStyle {
    IEEE,
    FN,
    FNUZ,
}

#[derive(Debug, Clone, Copy)]
pub struct F8<const E: u32, const M: u32,
    const N: NanStyle = {NanStyle::IEEE},
    const B: i32 = {(1 << (E - 1)) - 1},
>(u8);

#[derive(Debug, Clone, Copy)]
pub struct F16<const E: u32, const M: u32>(u16);

#[allow(non_camel_case_types)]
pub type f16 = F16<5, 10>;

#[allow(non_camel_case_types)]
pub type bf16 = F16<8, 7>;

fn fast_exp2(x: i32) -> f64 {
    f64::from_bits(match 0x3FF + x {
        0x800.. => 0x7FF << 52,
        #[allow(clippy::cast_sign_loss)]
        s@1..=0x7FF => (s as u64) << 52,
        s@ -51..=0 => 1 << (51 + s),
        _ => 0,
    })
}

macro_rules! define_round_for_mantissa {
    ($name:ident, $f:ty) => {
        fn $name<const M: u32>(x: $f) -> $f {
            let x = x.to_bits();
            let shift = <$f>::MANTISSA_DIGITS - 1 - M;
            let ulp = 1 << shift;
            let bias = (ulp >> 1) - (!(x >> shift) & 1);
            <$f>::from_bits((x + bias) & !(ulp - 1))
        }
    };
}

define_round_for_mantissa!(round_f32_for_mantissa, f32);
define_round_for_mantissa!(round_f64_for_mantissa, f64);

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> F8<E, M, N, B> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 8);
    const _HAS_EXPONENT: () = assert!(E > 0);

    pub const RADIX: u32 = 2;
    pub const MANTISSA_DIGITS: u32 = M + 1;
    pub const MAX_EXP: i32 = (1 << E) - B - matches!(N, NanStyle::IEEE) as i32;
    pub const MIN_EXP: i32 = 2 - B;

    const _IS_LOSSLESS_INTO_F32: () = assert!(
        Self::MAX_EXP <= f32::MAX_EXP &&
        Self::MIN_EXP >= f32::MIN_EXP
    );

    pub const NAN: Self = Self(match N {
        NanStyle::IEEE => ((1 << (E + 1)) - 1) << (M - 1),
        NanStyle::FN   => (1 << (E + M)) - 1,
        NanStyle::FNUZ =>  1 << (E + M),
    });

    pub const HUGE: Self = Self(match N {
        NanStyle::IEEE => ((1 << E) - 1) << M,
        NanStyle::FN   => (1 << (E + M)) - 2,
        NanStyle::FNUZ => (1 << (E + M)) - 1,
    });

    pub const MAX: Self = Self(Self::HUGE.0 - matches!(N, NanStyle::IEEE) as u8);
    pub const TINY: Self = Self(1);
    pub const MIN_POSITIVE: Self = Self(1 << M);
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    const ABS_MASK: u8 = (1 << (E + M)) - 1;

    #[must_use]
    pub const fn from_bits(v: u8) -> Self {
        let mask = if E + M >= 7 { 0xFF } else { (1 << (E + M + 1)) - 1 };
        Self(v & mask)
    }

    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn is_nan(self) -> bool {
        match N {
            NanStyle::IEEE => self.0 & Self::ABS_MASK > Self::HUGE.0,
            NanStyle::FN   => self.0 & Self::ABS_MASK == Self::NAN.0,
            NanStyle::FNUZ => self.0 == Self::NAN.0,
        }
    }

    #[must_use]
    pub const fn is_infinite(self) -> bool {
        matches!(N, NanStyle::IEEE) && self.0 & Self::ABS_MASK == Self::HUGE.0
    }

    #[must_use]
    pub const fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    #[must_use]
    pub const fn is_sign_positive(self) -> bool {
        self.0 >> (E + M) & 1 == 0
    }

    #[must_use]
    pub const fn is_sign_negative(self) -> bool {
        self.0 >> (E + M) & 1 == 1
    }
}

impl<const E: u32, const M: u32, const B: i32> F8<E, M, {NanStyle::IEEE}, B> {
    pub const INFINITY: Self = Self(((1 << E) - 1) << M);
    pub const NEG_INFINITY: Self = Self(Self::INFINITY.0 | 1 << (E + M));
}

impl<const E: u32, const M: u32> F16<E, M> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 16);
    const _HAS_EXPONENT: () = assert!(E > 0);

    pub const RADIX: u32 = 2;
    pub const MANTISSA_DIGITS: u32 = M + 1;
    pub const MAX_EXP: i32 = 1 << (E - 1);
    pub const MIN_EXP: i32 = 3 - Self::MAX_EXP;
    pub const INFINITY: Self = Self(((1 << E) - 1) << M);
    pub const NAN: Self = Self(((1 << (E + 1)) - 1) << (M - 1));
    pub const HUGE: Self = Self::INFINITY;
    pub const MAX: Self = Self(Self::INFINITY.0 - 1);
    pub const TINY: Self = Self(1);
    pub const MIN_POSITIVE: Self = Self(1 << M);
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    const _IS_LOSSLESS_INTO_F32: () = assert!(
        Self::MAX_EXP <= f32::MAX_EXP &&
        Self::MIN_EXP >= f32::MIN_EXP
    );

    const ABS_MASK: u16 = (1 << (E + M)) - 1;

    #[must_use]
    pub const fn from_bits(v: u16) -> Self {
        let mask = if E + M >= 15 { 0xFFFF } else { (1 << (E + M + 1)) - 1 };
        Self(v & mask)
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[must_use]
    pub const fn is_nan(self) -> bool {
        self.0 & Self::ABS_MASK > Self::INFINITY.0
    }

    #[must_use]
    pub const fn is_infinite(self) -> bool {
        self.0 & Self::ABS_MASK == Self::INFINITY.0
    }

    #[must_use]
    pub const fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    #[must_use]
    pub const fn is_sign_positive(self) -> bool {
        self.0 >> (E + M) & 1 == 0
    }

    #[must_use]
    pub const fn is_sign_negative(self) -> bool {
        self.0 >> (E + M) & 1 == 1
    }
}

pub trait Underlying<T>: Copy {
    fn from_bits(v: T) -> Self;
    fn to_bits(self) -> T;
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> Underlying<u8> for F8<E, M, N, B> {
    fn from_bits(v: u8) -> Self {
        Self::from_bits(v)
    }

    fn to_bits(self) -> u8 {
        self.to_bits()
    }
}

impl<const E: u32, const M: u32> Underlying<u16> for F16<E, M> {
    fn from_bits(v: u16) -> Self {
        Self::from_bits(v)
    }

    fn to_bits(self) -> u16 {
        self.to_bits()
    }
}

macro_rules! define_into_f32 {
    ($name:ident, $f:ty) => {
        fn $name(x: $f) -> f32 {
            let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
            let magnitude = x.to_bits() & <$f>::ABS_MASK;

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

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32>
From<F8<E, M, N, B>> for f32 {
    define_into_f32!(from, F8<E, M, N, B>);
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32>
From<F8<E, M, N, B>> for f64 {
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
        let a = self.to_bits();
        let b = other.to_bits();
        let eq = a == b && !self.is_nan();
        eq || !matches!(N, NanStyle::FNUZ) && (a | b) & Self::ABS_MASK == 0
    }
}

impl<const E: u32, const M: u32> PartialEq for F16<E, M> {
    fn eq(&self, other: &Self) -> bool {
        let a = self.to_bits();
        let b = other.to_bits();
        let eq = a == b && !self.is_nan();
        eq || (a | b) & Self::ABS_MASK == 0
    }
}

pub trait Minifloat: Copy + PartialEq {
    const E: u32;
    const M: u32;
    const N: NanStyle = NanStyle::IEEE;
    const B: i32 = (1 << (Self::E - 1)) - 1;

    fn from_f32(x: f32) -> Self;
    fn from_f64(x: f64) -> Self;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool { !self.is_nan() && !self.is_infinite() }
    fn is_sign_positive(self) -> bool { !self.is_sign_negative() }
    fn is_sign_negative(self) -> bool;
}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32> Minifloat for F8<E, M, N, B> {
    const E: u32 = E;
    const M: u32 = M;
    const N: NanStyle = N;
    const B: i32 = B;

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    fn from_f32(x: f32) -> Self {
        let bits = round_f32_for_mantissa::<M>(x).to_bits();
        let sign_bit = ((bits >> 31) as u8) << (E + M);

        if x.is_nan() {
            return Self(Self::NAN.0 | sign_bit);
        }

        let diff = (Self::MIN_EXP - f32::MIN_EXP) << M;
        let magnitude = bits << 1 >> (f32::MANTISSA_DIGITS - M);
        let magnitude = magnitude as i32 - diff;

        if magnitude < 1 << M {
            let ticks = f64::from(x.abs()) * fast_exp2(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ticks = ticks.round_ties_even() as u8;
            return Self((u8::from(N != NanStyle::FNUZ || ticks != 0) * sign_bit) | ticks);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self(magnitude.min(i32::from(Self::HUGE.0)) as u8 | sign_bit)
    }
    
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    fn from_f64(x: f64) -> Self {
        let bits = round_f64_for_mantissa::<M>(x).to_bits();
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

    fn is_nan(self) -> bool { self.is_nan() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn is_sign_positive(self) -> bool { self.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.is_sign_negative() }
}

impl<const E: u32, const M: u32> Minifloat for F16<E, M> {
    const E: u32 = E;
    const M: u32 = M;

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    fn from_f32(x: f32) -> Self {
        let bits = round_f32_for_mantissa::<M>(x).to_bits();
        let sign_bit = ((bits >> 31) as u16) << (E + M);

        if x.is_nan() {
            return Self(Self::NAN.0 | sign_bit);
        }

        let diff = (Self::MIN_EXP - f32::MIN_EXP) << M;
        let magnitude = bits << 1 >> (f32::MANTISSA_DIGITS - M);
        let magnitude = magnitude as i32 - diff;

        if magnitude < 1 << M {
            let ticks = f64::from(x.abs()) * fast_exp2(Self::MANTISSA_DIGITS as i32 - Self::MIN_EXP);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            return Self(ticks.round_ties_even() as u16 | sign_bit);
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Self(magnitude.min(i32::from(Self::HUGE.0)) as u16 | sign_bit)
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    fn from_f64(x: f64) -> Self {
        let bits = round_f64_for_mantissa::<M>(x).to_bits();
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

    fn is_nan(self) -> bool { self.is_nan() }
    fn is_infinite(self) -> bool { self.is_infinite() }
    fn is_sign_positive(self) -> bool { self.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.is_sign_negative() }
}