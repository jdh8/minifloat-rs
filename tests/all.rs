use core::fmt::Debug;
use minifloat::example::*;
use minifloat::{minifloat, Minifloat, NanStyle};

minifloat!(struct F8E2M5(u8): 2, 5);
minifloat!(struct F8E2M5FN(u8): 2, 5, FN);
minifloat!(struct F8E2M5FNUZ(u8): 2, 5, FNUZ);

minifloat!(struct F8E3M4FNUZ(u8): 3, 4, FNUZ);
minifloat!(struct F8E5M2FN(u8): 5, 2, FN);

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

fn for_all<T: Minifloat<Bits = u8>>(f: impl Fn(T) -> bool) -> bool {
    (0..=u8::MAX).map(T::from_bits).all(f)
}

fn check_equality<T: Minifloat<Bits = u8> + Debug>() -> bool {
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

#[test]
fn test_equality() {
    assert!(check_equality::<F8E2M5>());
    assert!(check_equality::<F8E2M5FN>());
    assert!(check_equality::<F8E2M5FNUZ>());

    assert!(check_equality::<F8E3M4>());
    assert!(check_equality::<F8E3M4FN>());
    assert!(check_equality::<F8E3M4FNUZ>());

    assert!(check_equality::<F8E4M3>());
    assert!(check_equality::<F8E4M3FN>());
    assert!(check_equality::<F8E4M3FNUZ>());

    assert!(check_equality::<F8E4M3B11>());
    assert!(check_equality::<F8E4M3B11FN>());
    assert!(check_equality::<F8E4M3B11FNUZ>());

    assert!(check_equality::<F8E5M2>());
    assert!(check_equality::<F8E5M2FN>());
    assert!(check_equality::<F8E5M2FNUZ>());
}
