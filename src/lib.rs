// This file is part of the minifloat project.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#![no_std]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use core::marker::ConstParamTy;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, ConstParamTy, PartialEq, Eq)]
pub enum NanStyle {
    IEEE,
    FN,
    FNUZ,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F8<const E: u32, const M: u32,
    const N: NanStyle = {NanStyle::IEEE},
    const B: i32 = {(1 << (E - 1)) - 1},
>(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16<const E: u32, const M: u32>(u16);

impl <const E: u32, const M: u32, const N: NanStyle, const B: i32> F8<E, M, N, B> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 8);
    const _HAS_EXPONENT: () = assert!(E > 0);
    const _HAS_SIGNIFICAND: () = assert!(M > 0);

    pub const RADIX: u32 = 2;
    pub const MANTISSA_DIGITS: u32 = M + 1;
    pub const MAX_EXP: i32 = (1 << E) - B - matches!(N, NanStyle::IEEE) as i32;
    pub const MIN_EXP: i32 = 2 - B;

    pub const NAN: Self = Self(match N {
        NanStyle::IEEE => ((1 << (E + 1)) - 1) << (M - 1),
        NanStyle::FN   => (1 << (E + M)) - 1,
        NanStyle::FNUZ =>  1 << (E + M),
    });

    pub const MAX: Self = Self(match N {
        NanStyle::IEEE => (((1 << E) - 1) << M) - 1,
        NanStyle::FN   => (1 << (E + M)) - 2,
        NanStyle::FNUZ => (1 << (E + M)) - 1,
    });

    pub const MIN_POSITIVE: Self = Self(1 << M);
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    #[must_use]
    pub const fn from_bits(v: u8) -> Self {
        Self(v)
    }

    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }
}

impl <const E: u32, const M: u32, const B: i32> F8<E, M, {NanStyle::IEEE}, B> {
    pub const INFINITY: Self = Self(((1 << E) - 1) << M);
    pub const NEG_INFINITY: Self = Self(Self::INFINITY.0 | 1 << (E + M));
}

impl <const E: u32, const M: u32> F16<E, M> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 16);
    const _HAS_EXPONENT: () = assert!(E > 0);
    const _HAS_SIGNIFICAND: () = assert!(M > 0);

    pub const RADIX: u32 = 2;
    pub const MANTISSA_DIGITS: u32 = M + 1;
    pub const MAX_EXP: i32 = 1 << (E - 1);
    pub const MIN_EXP: i32 = 3 - Self::MAX_EXP;
    pub const INFINITY: Self = Self(((1 << E) - 1) << M);
    pub const NAN: Self = Self(((1 << (E + 1)) - 1) << (M - 1));
    pub const MAX: Self = Self((((1 << E) - 1) << M) - 1);
    pub const MIN_POSITIVE: Self = Self(1 << M);
    pub const MIN: Self = Self(Self::MAX.0 | 1 << (E + M));

    #[must_use]
    pub const fn from_bits(v: u16) -> Self {
        Self(v)
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }
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

enum Assert<const CHECK: bool> {}
trait IsTrue {}
impl IsTrue for Assert<true> {}

impl<const E: u32, const M: u32, const N: NanStyle, const B: i32>
From<F8<E, M, N, B>> for f32
where
    Assert<{F8::<E, M, N, B>::MAX_EXP <= Self::MAX_EXP}>: IsTrue,
    Assert<{F8::<E, M, N, B>::MIN_EXP >= Self::MIN_EXP}>: IsTrue,
{
    fn from(_: F8<E, M, N, B>) -> Self {
        todo!()
    }
}