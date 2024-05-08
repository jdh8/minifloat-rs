// This file is part of the minifloat project.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#![cfg(test)]
#![allow(clippy::float_cmp)]

use crate::NanStyle::{FN, FNUZ};

#[test]
fn test_exp2() {
    (-1200..1200).for_each(|x| assert_eq!(crate::fast_exp2(x), f64::from(x).exp2()));
}

#[allow(clippy::cast_sign_loss)]
fn test_finite_bits_f8<const E: u32, const M: u32>(x: f32, bits: u8)
where [(); {(1 << (E - 1)) - 1} as usize]: /* Rust is baka! */ {
    assert_eq!(crate::F8::<E, M>::from_f32(x).to_bits(), bits);
    assert_eq!(crate::F8::<E, M, {FN}>::from_f32(x).to_bits(), bits);
    assert_eq!(crate::F8::<E, M, {FNUZ}>::from_f32(x).to_bits(), bits);
}

#[test]
#[allow(clippy::unusual_byte_groupings)]
fn test_finite_bits() {
    test_finite_bits_f8::<3, 4>(2.0, 0x40);
    test_finite_bits_f8::<4, 3>(2.0, 0x40);
    test_finite_bits_f8::<5, 2>(2.0, 0x40);
    //assert_eq!(crate::F16::<5, 7>::from_f32(2.0).to_bits(), 0b0_10000_0000000);

    test_finite_bits_f8::<3, 4>(1.0, 0b0_011_0000);
    test_finite_bits_f8::<4, 3>(1.0, 0b0_0111_000);
    test_finite_bits_f8::<5, 2>(1.0, 0b0_01111_00);
    //assert_eq!(crate::F16::<5, 7>::from_f32(1.0).to_bits(), 0b0_01111_0000000);

    test_finite_bits_f8::<3, 4>(-1.25, 0b1_011_0100);
    test_finite_bits_f8::<4, 3>(-1.25, 0b1_0111_010);
    test_finite_bits_f8::<5, 2>(-1.25, 0b1_01111_01);
    //assert_eq!(crate::F16::<5, 7>::from_f32(-1.25).to_bits(), 0b1_01111_0100000);
}

fn test_identity_conversion_f8<T: crate::Minifloat + crate::Underlying<u8>>()
where f32: From<T> {
    (0..=0xFF).for_each(|i: u8| {
        let x = T::from_bits(i);
        let y = T::from_f32(f32::from(x));
        assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
    });
}

#[test]
fn identity_conversion() {
    test_identity_conversion_f8::<crate::F8<3, 4>>();
    test_identity_conversion_f8::<crate::F8<4, 3>>();
    test_identity_conversion_f8::<crate::F8<5, 2>>();
}