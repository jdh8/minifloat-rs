// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::support::*;
use minifloat::{
    minifloat, Minifloat, BF16, F16, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
    F8E4M3B11FNUZ, F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
};

use core::fmt::Debug;
use core::hash::Hash;

#[test]
fn test_to_f32() {
    struct CheckToF32;
    impl Check for CheckToF32 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            assert!(same_f32(T::ZERO.to_f32(), 0.0));
            assert!(same_f32(
                (-T::ZERO).to_f32(),
                if T::HAS_NEG_ZERO { -0.0 } else { 0.0 }
            ));
            // A type wider than `f32` cannot round-trip through it; the
            // encoder is checked against a reference instead.
            !T::HAS_EXACT_F32_CONVERSION
                || for_all::<T>(|x| same_mini(T::from_f32(x.to_f32()), x))
        }
    }
    test_most_8_bits(CheckToF32);
    test_16_bits(CheckToF32);
}

#[test]
fn test_to_f64() {
    struct CheckToF64;
    impl Check for CheckToF64 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            assert!(same_f64(T::ZERO.to_f64(), 0.0));
            assert!(same_f64(
                (-T::ZERO).to_f64(),
                if T::HAS_NEG_ZERO { -0.0 } else { 0.0 }
            ));
            // See [`test_to_f32`]: the same caveat applies one size up.
            !T::HAS_EXACT_F64_CONVERSION
                || for_all::<T>(|x| same_mini(T::from_f64(x.to_f64()), x))
        }
    }
    test_most_8_bits(CheckToF64);
    test_16_bits(CheckToF64);
}

#[test]
fn test_to_floats() {
    struct CheckToFloats;
    impl Check for CheckToFloats {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            !T::HAS_EXACT_F32_CONVERSION
                || for_all::<T>(|x| same_f64(x.to_f32().into(), x.to_f64()))
        }
    }
    test_most_8_bits(CheckToFloats);
    test_16_bits(CheckToFloats);
}

#[test]
fn test_has_exact_conversion_consts() {
    const _: () = assert!(F16::HAS_EXACT_F32_CONVERSION);
    const _: () = assert!(F16::HAS_EXACT_F64_CONVERSION);
    const _: () = assert!(BF16::HAS_EXACT_F32_CONVERSION);
    const _: () = assert!(BF16::HAS_EXACT_F64_CONVERSION);
    const _: () = assert!(F8E4M3FN::HAS_EXACT_F32_CONVERSION);
    const _: () = assert!(F8E4M3FN::HAS_EXACT_F64_CONVERSION);
}

#[test]
fn test_from_lossless() {
    fn check<T: Minifloat + Debug + Into<f32> + Into<f64>>() -> bool
    where
        T::Bits: TryFrom<Mask>,
    {
        for_all::<T>(|x| {
            let via_from_f32: f32 = x.into();
            let via_from_f64: f64 = x.into();
            same_f32(via_from_f32, x.to_f32()) && same_f64(via_from_f64, x.to_f64())
        })
    }
    assert!(check::<F16>());
    assert!(check::<BF16>());
    assert!(check::<F8E4M3FN>());
    assert!(check::<F8E5M2>());
    assert!(check::<F8E4M3FNUZ>());
    assert!(check::<F4E2M1FN>());
    assert!(check::<F6E2M3FN>());
    assert!(check::<F6E3M2FN>());
    assert!(check::<F8E3M4>());
    assert!(check::<F8E4M3>());
    assert!(check::<F8E4M3B11FNUZ>());
    assert!(check::<F8E5M2FNUZ>());
}

#[test]
fn test_integer_decode_reconstruction() {
    struct CheckIntegerDecode;
    impl Check for CheckIntegerDecode {
        fn check<T: Minifloat + Debug + Hash>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for_all::<T>(|x| {
                let (mantissa, exponent, sign) = x.integer_decode();
                if x.is_nan() {
                    return (mantissa, exponent, sign) == (0, 0, 0);
                }
                if sign != 1 && sign != -1 {
                    return false;
                }
                // Reconstruction is only guaranteed for finite values, matching
                // `f32::integer_decode`.
                if !x.is_finite() {
                    return true;
                }
                #[allow(clippy::cast_precision_loss)]
                let reconstructed = f64::from(sign) * scaled(mantissa as f64, i32::from(exponent));
                same_f64(reconstructed, x.to_f64())
            })
        }
    }
    test_most_8_bits(CheckIntegerDecode);
    test_16_bits(CheckIntegerDecode);
}

/// The wide shapes where the exactness constants must flip
///
/// A wrongly-false constant would make the gated round-trips above pass
/// vacuously, so the shapes on either side of each boundary are pinned here.
#[test]
fn test_has_exact_conversion_consts_wide() {
    minifloat!(struct E11M4(u16): 11, 4);
    minifloat!(struct E12M3(u16): 12, 3);
    minifloat!(struct E2M13(u16): 2, 13);

    const _: () = assert!(!E11M4::HAS_EXACT_F32_CONVERSION);
    const _: () = assert!(E11M4::HAS_EXACT_F64_CONVERSION);
    const _: () = assert!(!E12M3::HAS_EXACT_F32_CONVERSION);
    const _: () = assert!(!E12M3::HAS_EXACT_F64_CONVERSION);
    const _: () = assert!(E2M13::HAS_EXACT_F32_CONVERSION);
    const _: () = assert!(E2M13::HAS_EXACT_F64_CONVERSION);
}
