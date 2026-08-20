// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::support::*;
use minifloat::{Minifloat, F16};

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::{BuildHasher, Hash};
use core::num::FpCategory;

#[test]
fn test_eq() {
    struct CheckEq;
    impl Check for CheckEq {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            let fixed_point = if T::M == 0 { 2.0 } else { 3.0 };
            assert!(same_f32(T::from_f32(fixed_point).to_f32(), fixed_point));

            let fixed_point = f64::from(fixed_point);
            assert!(same_f64(T::from_f64(fixed_point).to_f64(), fixed_point));

            assert!(T::ZERO.to_bits() == narrow::<T>(0));
            assert_eq!(T::ZERO, T::from_f32(0.0));
            assert_eq!(T::ZERO, T::from_f32(-0.0));

            assert_eq!(same_mini(T::ZERO, T::from_f32(-0.0)), !T::HAS_NEG_ZERO);

            match T::NAN {
                Some(nan) => {
                    assert!(nan.is_nan());
                    assert!(T::from_f32(f32::NAN).is_nan());
                    assert!(T::from_f64(f64::NAN).is_nan());

                    assert!(nan.ne(&nan));
                    assert!(same_mini(nan, nan));
                }
                // Without a NaN encoding, a NaN input saturates to the maximum
                // finite value with the sign preserved.
                None => {
                    assert!(same_mini(T::from_f32(f32::NAN), T::MAX));
                    assert!(same_mini(T::from_f64(f64::NAN), T::MAX));
                    assert!(same_mini(T::from_f32(-f32::NAN), T::MIN));
                    assert!(same_mini(T::from_f64(-f64::NAN), T::MIN));

                    // An all-ones payload carries out of the exponent field if
                    // it ever reaches the rounding path.
                    assert!(same_mini(T::from_f32(f32::from_bits(0x7FFF_FFFF)), T::MAX));
                    assert!(same_mini(T::from_f32(f32::from_bits(0xFFFF_FFFF)), T::MIN));
                    assert!(same_mini(
                        T::from_f64(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF)),
                        T::MAX
                    ));
                    assert!(same_mini(
                        T::from_f64(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)),
                        T::MIN
                    ));
                }
            }

            for_all::<T>(|x| x.ne(&x) == x.is_nan())
        }
    }
    test_most_8_bits(CheckEq);
    test_16_bits(CheckEq);
}

#[test]
fn test_neg() {
    struct CheckNeg;
    impl Check for CheckNeg {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for_all::<T>(|x| x.to_bits() == (-(-x)).to_bits())
        }
    }
    test_most_8_bits(CheckNeg);
    test_16_bits(CheckNeg);
}

#[test]
fn test_partial_cmp() {
    struct CheckOrd;
    impl Check for CheckOrd {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for_all::<T>(|x| {
                for_all::<T>(|y| x.partial_cmp(&y) == x.to_f32().partial_cmp(&y.to_f32()))
            })
        }
    }
    test_most_8_bits(CheckOrd);
}

#[test]
fn test_classify() {
    struct CheckClassify;
    impl Check for CheckClassify {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for_all::<T>(|x| {
                u32::from(x.is_nan()) << FpCategory::Nan as u8
                    | u32::from(x.is_infinite()) << FpCategory::Infinite as u8
                    | u32::from(x.is_normal()) << FpCategory::Normal as u8
                    | u32::from(x.is_subnormal()) << FpCategory::Subnormal as u8
                    | u32::from(x == T::from_bits(narrow::<T>(0))) << FpCategory::Zero as u8
                    == 1 << x.classify() as u8
            })
        }
    }
    test_most_8_bits(CheckClassify);
    test_16_bits(CheckClassify);
}

#[test]
fn test_const_comparison_helpers() {
    const _: () = {
        let zero = F16::ZERO;
        let one = F16::from_bits(0x3C00);
        assert!(zero.const_eq(F16::ZERO));
        assert!(!zero.const_eq(one));
        assert!(matches!(
            one.const_partial_cmp(zero),
            Some(Ordering::Greater)
        ));
        assert!(F16::NAN.const_partial_cmp(zero).is_none());
        assert!(F16::NAN.is_nan());
        assert!(F16::INFINITY.const_eq(F16::HUGE));
    };
}

#[test]
fn test_hash_consistent_with_eq() {
    struct CheckHash;
    impl Check for CheckHash {
        fn check<T: Minifloat + Debug + Hash>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            let state = std::collections::hash_map::RandomState::new();
            for_all::<T>(|x| {
                for_all::<T>(|y| x != y || state.hash_one(x) == state.hash_one(y))
            })
        }
    }
    test_most_8_bits(CheckHash);
}
