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
use crate::{Minifloat, Transmute, F16, F8};
use core::fmt::Debug;

macro_rules! for_each_type {
    ($f8:ident, $f16:ident) => {
        $f8::<F8<3, 4>>();
        $f8::<F8<3, 4, { FN }>>();
        $f8::<F8<3, 4, { FNUZ }>>();

        $f8::<F8<4, 3>>();
        $f8::<F8<4, 3, { FN }>>();
        $f8::<F8<4, 3, { FNUZ }>>();

        $f8::<F8<5, 2>>();
        $f8::<F8<5, 2, { FN }>>();
        $f8::<F8<5, 2, { FNUZ }>>();

        $f16::<F16<5, 7>>();
        $f16::<crate::f16>();
        $f16::<crate::bf16>();
    };
}

macro_rules! are_equivalent {
    ($x:expr, $y:expr) => {
        $x.to_bits() == $y.to_bits() || ($x.is_nan() && $y.is_nan())
    };
}

#[test]
fn test_exp2() {
    (-1200..1200).for_each(|x| assert_eq!(crate::fast_exp2(x), f64::from(x).exp2()));
}

struct BiasConstant<const N: i32>;

#[allow(clippy::cast_sign_loss)]
fn test_finite_bits_f8<const E: u32, const M: u32>(x: f32, bits: u8)
where
    BiasConstant<{ (1 << (E - 1)) - 1 }>:,
    F8<E, M>: Minifloat,
    F8<E, M, { FN }>: Minifloat,
    F8<E, M, { FNUZ }>: Minifloat,
{
    assert_eq!(F8::<E, M>::from_f32(x).to_bits(), bits);
    assert_eq!(F8::<E, M, { FN }>::from_f32(x).to_bits(), bits);
    assert_eq!(F8::<E, M, { FNUZ }>::from_f32(x).to_bits(), bits);
}

#[test]
#[allow(clippy::unusual_byte_groupings)]
fn test_finite_bits() {
    test_finite_bits_f8::<3, 4>(2.0, 0x40);
    test_finite_bits_f8::<4, 3>(2.0, 0x40);
    test_finite_bits_f8::<5, 2>(2.0, 0x40);
    assert_eq!(F16::<5, 7>::from_f32(2.0).to_bits(), 0b0_10000_0000000);
    assert_eq!(crate::f16::from_f32(2.0).to_bits(), 0x4000);
    assert_eq!(crate::bf16::from_f32(2.0).to_bits(), 0x4000);

    test_finite_bits_f8::<3, 4>(1.0, 0b0_011_0000);
    test_finite_bits_f8::<4, 3>(1.0, 0b0_0111_000);
    test_finite_bits_f8::<5, 2>(1.0, 0b0_01111_00);
    assert_eq!(F16::<5, 7>::from_f32(1.0).to_bits(), 0b0_01111_0000000);
    assert_eq!(crate::f16::from_f32(1.0).to_bits(), 0b0_01111_00000_00000);
    assert_eq!(crate::bf16::from_f32(1.0).to_bits(), 0b0_0111_1111_0000000);

    test_finite_bits_f8::<3, 4>(-1.25, 0b1_011_0100);
    test_finite_bits_f8::<4, 3>(-1.25, 0b1_0111_010);
    test_finite_bits_f8::<5, 2>(-1.25, 0b1_01111_01);
    assert_eq!(F16::<5, 7>::from_f32(-1.25).to_bits(), 0b1_01111_0100000);
    assert_eq!(crate::f16::from_f32(-1.25).to_bits(), 0b1_01111_01000_00000);
    assert_eq!(
        crate::bf16::from_f32(-1.25).to_bits(),
        0b1_0111_1111_0100000
    );
}

fn test_epsilon_f8<T: Minifloat + Transmute<u8> + Debug>()
where
    f32: From<T>,
{
    let bits = T::from_f32(1.0).to_bits();
    let next_up = T::from_bits(bits + 1);
    let next_down = T::from_bits(bits - 1);

    assert_eq!(f32::from(next_up) - 1.0, T::EPSILON.into());
    assert_eq!(2.0 * (1.0 - f32::from(next_down)), T::EPSILON.into());
}

fn test_epsilon_f16<T: Minifloat + Transmute<u16> + Debug>()
where
    f32: From<T>,
{
    let bits = T::from_f32(1.0).to_bits();
    let next_up = T::from_bits(bits + 1);
    let next_down = T::from_bits(bits - 1);

    assert_eq!(f32::from(next_up) - 1.0, T::EPSILON.into());
    assert_eq!(2.0 * (1.0 - f32::from(next_down)), T::EPSILON.into());
}

#[test]
fn test_epsilon() {
    for_each_type!(test_epsilon_f8, test_epsilon_f16);
}

fn test_equality_f8<T: Minifloat + Transmute<u8> + Debug>()
where
    f32: From<T>,
    f64: From<T>,
{
    assert_eq!(f32::from(T::from_f32(-3.0)), -3.0);
    assert_eq!(f64::from(T::from_f64(-3.0)), -3.0);
    assert_eq!(T::from_f32(0.0), T::from_f32(-0.0));
    assert_eq!(
        T::from_f32(0.0).to_bits() == T::from_f32(-0.0).to_bits(),
        T::N == FNUZ
    );
    assert!(T::from_f32(f32::NAN).is_nan());
    assert!(f32::from(T::from_f32(f32::NAN)).is_nan());
    assert!(f64::from(T::from_f64(f64::NAN)).is_nan());

    (0..=0xFF).map(T::from_bits).for_each(|x| {
        assert_eq!(x.ne(&x), x.is_nan());
    });
}

fn test_equality_f16<T: Minifloat + Transmute<u16> + Debug>()
where
    f32: From<T>,
    f64: From<T>,
{
    assert_eq!(f32::from(T::from_f32(-3.0)), -3.0);
    assert_eq!(f64::from(T::from_f64(-3.0)), -3.0);
    assert_eq!(T::from_f32(0.0), T::from_f32(-0.0));
    assert_ne!(
        T::from_f32(0.0).to_bits(),
        T::from_f32(-0.0).to_bits(),
        "{}",
        core::any::type_name::<T>()
    );
    assert!(T::from_f32(f32::NAN).is_nan());
    assert!(f32::from(T::from_f32(f32::NAN)).is_nan());
    assert!(f64::from(T::from_f64(f64::NAN)).is_nan());

    (0..=0xFFFF).map(T::from_bits).for_each(|x| {
        assert_eq!(x.ne(&x), x.is_nan());
    });
}

#[test]
fn test_equality() {
    for_each_type!(test_equality_f8, test_equality_f16);
}

fn test_comparison_f8<T: Minifloat + Transmute<u8> + Debug>()
where
    f32: From<T>,
{
    (0..=0xFF).map(T::from_bits).for_each(|x| {
        (0..=0xFF).map(T::from_bits).for_each(|y| {
            assert_eq!(x.partial_cmp(&y), f32::from(x).partial_cmp(&f32::from(y)));
        });
    });
}

fn test_comparison_f16<T: Minifloat + Transmute<u16> + Debug>()
where
    f32: From<T>,
{
    (0..=0xFFFF)
        .step_by(17 << (T::E + T::M) >> 15 | 1)
        .map(T::from_bits)
        .for_each(|x| {
            (0..=0xFFFF)
                .step_by(19 << (T::E + T::M) >> 15 | 1)
                .map(T::from_bits)
                .for_each(|y| {
                    assert_eq!(x.partial_cmp(&y), f32::from(x).partial_cmp(&f32::from(y)));
                });
        });
}

#[test]
fn test_comparison() {
    for_each_type!(test_comparison_f8, test_comparison_f16);
}

fn test_neg_f8<T: Minifloat + Transmute<u8> + Debug>()
where
    f32: From<T>,
{
    (0..=0xFF).map(T::from_bits).for_each(|x| {
        let y = T::from_f32(-f32::from(x));
        assert!(are_equivalent!(y, -x), "{y:?} is not {:?}", -x);
    });
}

fn test_neg_f16<T: Minifloat + Transmute<u16> + Debug>()
where
    f32: From<T>,
{
    (0..=0xFFFF).map(T::from_bits).for_each(|x| {
        let y = T::from_f32(-f32::from(x));
        assert!(are_equivalent!(y, -x), "{y:?} is not {:?}", -x);
    });
}

#[test]
fn test_neg() {
    for_each_type!(test_neg_f8, test_neg_f16);
}

fn test_identity_conversion_f8<T: Minifloat + Transmute<u8> + Debug>()
where
    f32: From<T>,
    f64: From<T>,
{
    (0..=0xFF).map(T::from_bits).for_each(|x| {
        let y = T::from_f32(f32::from(x));
        assert!(are_equivalent!(x, y), "{x:?} is not {y:?}");

        let y = T::from_f64(f64::from(x));
        assert!(are_equivalent!(x, y), "{x:?} is not {y:?}");
    });
}

fn test_identity_conversion_f16<T: Minifloat + Transmute<u16> + Debug>()
where
    f32: From<T>,
    f64: From<T>,
{
    (0..=0xFFFF).map(T::from_bits).for_each(|x| {
        let y = T::from_f32(f32::from(x));
        assert!(are_equivalent!(x, y), "{x:?} is not {y:?}");

        let y = T::from_f64(f64::from(x));
        assert!(are_equivalent!(x, y), "{x:?} is not {y:?}");
    });
}

#[test]
fn test_identity_conversion() {
    for_each_type!(test_identity_conversion_f8, test_identity_conversion_f16);
}

fn test_huge_f8<T: Minifloat + Transmute<u8> + Debug>()
where
    f32: From<T>,
{
    let huge = T::HUGE;
    let nan = T::from_bits(huge.to_bits() + 1);
    let big = T::from_bits(huge.to_bits() - 1);

    assert!(huge > big);
    assert!(nan.is_nan());
}

fn test_huge_f16<T: Minifloat + Transmute<u16> + Debug>()
where
    f32: From<T>,
{
    let huge = T::HUGE;
    let nan = T::from_bits(huge.to_bits() + 1);
    let big = T::from_bits(huge.to_bits() - 1);

    assert!(huge > big);
    assert!(nan.is_nan());
}

#[test]
fn test_huge() {
    for_each_type!(test_huge_f8, test_huge_f16);
}