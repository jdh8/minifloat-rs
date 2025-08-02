// This file is part of the minifloat project.
//
// Copyright (C) 2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use core::fmt::Debug;
use minifloat::example::*;
use minifloat::{minifloat, Minifloat, NanStyle};
use num_traits::AsPrimitive;

minifloat!(struct F8E2M5(u8): 2, 5);
minifloat!(struct F8E2M5FN(u8): 2, 5, FN);
minifloat!(struct F8E2M5FNUZ(u8): 2, 5, FNUZ);

minifloat!(struct F8E3M4FNUZ(u8): 3, 4, FNUZ);
minifloat!(struct F8E5M2FN(u8): 5, 2, FN);

/// Bitmask returned by [`bit_mask`]
///
/// This type must be an unsigned integer.
type Mask = u64;

/// Create a bitmask of the given width
const fn bit_mask(width: u32) -> Mask {
    assert!(width <= Mask::BITS);

    if width == 0 {
        0
    } else {
        !0 >> (Mask::BITS - width)
    }
}

/// Test floating-point identity like Object.is in JavaScript
///
/// This is necessary because NaN != NaN in C++.  We also want to differentiate
/// -0 from +0.  Using this functor, NaNs are considered identical to each
/// other, while +0 and -0 are considered different.
const fn same_f32(x: f32, y: f32) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

/// Test floating-point identity like Object.is in JavaScript
///
/// See also [`same_f32`].
const fn same_f64(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

/// Test floating-point identity like Object.is in JavaScript
///
/// See also [`same_f32`].
fn same_mini<T: Minifloat>(x: T, y: T) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

/// Iterate over all representations of a minifloat type
fn for_all<T: Minifloat>(f: impl Fn(T) -> bool) -> bool
where
    Mask: AsPrimitive<T::Bits>,
{
    (0..=bit_mask(T::BITWIDTH)).all(|bits| f(T::from_bits(bits.as_())))
}

/// Wrapper trait for checking properties of minifloats
///
/// This trait helps building generic test infrastructure.  Opposed to generic
/// functions, traits can work as parameters.
trait Check {
    /// Check properties of a minifloat type
    fn check<T: Minifloat + Debug>() -> bool
    where
        Mask: AsPrimitive<T::Bits>;

    /// Test typical minifloats
    fn test() {
        assert!(Self::check::<F8E2M5>());
        assert!(Self::check::<F8E2M5FN>());
        assert!(Self::check::<F8E2M5FNUZ>());

        assert!(Self::check::<F8E3M4>());
        assert!(Self::check::<F8E3M4FN>());
        assert!(Self::check::<F8E3M4FNUZ>());

        assert!(Self::check::<F8E4M3>());
        assert!(Self::check::<F8E4M3FN>());
        assert!(Self::check::<F8E4M3FNUZ>());

        assert!(Self::check::<F8E4M3B11>());
        assert!(Self::check::<F8E4M3B11FN>());
        assert!(Self::check::<F8E4M3B11FNUZ>());

        assert!(Self::check::<F8E5M2>());
        assert!(Self::check::<F8E5M2FN>());
        assert!(Self::check::<F8E5M2FNUZ>());
    }
}

struct CheckEquality;

impl Check for CheckEquality {
    fn check<T: Minifloat + Debug>() -> bool
    where
        Mask: AsPrimitive<T::Bits>,
    {
        let fixed_point = if T::M == 0 { 2.0 } else { 3.0 };
        assert!(same_f32(T::from_f32(fixed_point).to_f32(), fixed_point));

        let fixed_point = f64::from(fixed_point);
        assert!(same_f64(T::from_f64(fixed_point).to_f64(), fixed_point));

        assert_eq!(T::from_f32(0.0), T::from_f32(-0.0));
        assert_eq!(
            same_mini(T::from_f32(0.0), T::from_f32(-0.0)),
            T::N == NanStyle::FNUZ
        );

        assert!(T::NAN.is_nan());
        assert!(T::from_f32(f32::NAN).is_nan());
        assert!(T::from_f64(f64::NAN).is_nan());

        assert!(T::NAN.ne(&T::NAN));
        assert!(same_mini(T::NAN, T::NAN));

        for_all::<T>(|x| x.ne(&x) == x.is_nan())
    }
}

#[test]
fn test_equality() {
    CheckEquality::test();
}
