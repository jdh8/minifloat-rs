// This file is part of the minifloat project.
//
// Copyright (C) 2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use minifloat::{
    minifloat, Format, Minifloat, BF16, F16, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
    F8E4M3B11FNUZ, F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
};

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::{BuildHasher, Hash};
use core::num::FpCategory;

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

/// Narrow a bit pattern to a minifloat's storage
///
/// A minifloat is at most 16 bits wide, so the conversion always succeeds.
fn narrow<T: Minifloat>(bits: Mask) -> T::Bits
where
    T::Bits: TryFrom<Mask>,
{
    match bits.try_into() {
        Ok(narrowed) => narrowed,
        Err(_) => unreachable!("a minifloat is narrower than its storage"),
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

/// <var>significand</var> &times; 2<sup><var>exponent</var></sup>, correctly rounded
///
/// Splitting the scale keeps either factor within [`f64`]'s exponent range, so
/// the first product is exact and the second rounds at most once.  A single
/// `exp2` would flush to zero or to infinity long before the product does.
fn scaled(significand: f64, exponent: i32) -> f64 {
    let head = exponent.clamp(f64::MIN_EXP - 1, f64::MAX_EXP - 1);
    significand * f64::exp2(f64::from(head)) * f64::exp2(f64::from(exponent - head))
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
    T::Bits: TryFrom<Mask>,
{
    (0..=bit_mask(T::BITWIDTH)).all(|bits| f(T::from_bits(narrow::<T>(bits))))
}

/// Wrapper trait for checking properties of minifloats
///
/// This trait helps building generic test infrastructure.  Opposed to generic
/// functions, traits can work as parameters.
trait Check {
    /// Check properties of a minifloat type
    fn check<T: Minifloat + Debug + Hash>() -> bool
    where
        T::Bits: TryFrom<Mask>;
}

fn test_8_bits<T: Check>(_: T) {
    minifloat!(struct F8E2M5(u8): 2, 5);
    minifloat!(struct F8E2M5FN(u8): 2, 5, FN);
    minifloat!(struct F8E2M5FNUZ(u8): 2, 5, FNUZ);
    minifloat!(struct F8E2M5Finite(u8): 2, 5, Finite);

    // No longer shipped by the crate, but still worth exhaustive coverage.
    minifloat!(struct F8E3M4FN(u8): 3, 4, FN);
    minifloat!(struct F8E3M4FNUZ(u8): 3, 4, FNUZ);
    minifloat!(struct F8E4M3B11(u8): 4, 3, 11);
    minifloat!(struct F8E4M3B11FN(u8): 4, 3, 11, FN);

    minifloat!(struct F8E4M3Finite(u8): 4, 3, Finite);
    minifloat!(struct F8E5M2FN(u8): 5, 2, FN);
    minifloat!(struct F8E5M2Finite(u8): 5, 2, Finite);

    minifloat!(struct F8E6M1(u8): 6, 1);
    minifloat!(struct F8E6M1FN(u8): 6, 1, FN);
    minifloat!(struct F8E6M1FNUZ(u8): 6, 1, FNUZ);
    minifloat!(struct F8E6M1Finite(u8): 6, 1, Finite);

    assert!(T::check::<F8E2M5>());
    assert!(T::check::<F8E2M5FN>());
    assert!(T::check::<F8E2M5FNUZ>());
    assert!(T::check::<F8E2M5Finite>());

    assert!(T::check::<F8E3M4>());
    assert!(T::check::<F8E3M4FN>());
    assert!(T::check::<F8E3M4FNUZ>());

    assert!(T::check::<F8E4M3>());
    assert!(T::check::<F8E4M3FN>());
    assert!(T::check::<F8E4M3FNUZ>());
    assert!(T::check::<F8E4M3Finite>());

    assert!(T::check::<F8E4M3B11>());
    assert!(T::check::<F8E4M3B11FN>());
    assert!(T::check::<F8E4M3B11FNUZ>());

    assert!(T::check::<F8E5M2>());
    assert!(T::check::<F8E5M2FN>());
    assert!(T::check::<F8E5M2FNUZ>());
    assert!(T::check::<F8E5M2Finite>());

    assert!(T::check::<F8E6M1>());
    assert!(T::check::<F8E6M1FN>());
    assert!(T::check::<F8E6M1FNUZ>());
    assert!(T::check::<F8E6M1Finite>());
}

fn test_most_8_bits<T: Check>(x: T) {
    minifloat!(struct F6E2M3(u8): 2, 3);
    minifloat!(struct F6E2M3FNUZ(u8): 2, 3, FNUZ);

    minifloat!(struct F6E3M2(u8): 3, 2);
    minifloat!(struct F6E3M2FNUZ(u8): 3, 2, FNUZ);

    minifloat!(struct F4E2M1(u8): 2, 1);
    minifloat!(struct F4E2M1FNUZ(u8): 2, 1, FNUZ);

    // The crate's MX types carry the `Finite` format despite their `FN` names,
    // so these cover the `FN` format at the same shapes.
    minifloat!(struct E2M3FN(u8): 2, 3, FN);
    minifloat!(struct E3M2FN(u8): 3, 2, FN);
    minifloat!(struct E2M1FN(u8): 2, 1, FN);

    assert!(T::check::<F6E2M3>());
    assert!(T::check::<F6E2M3FN>());
    assert!(T::check::<F6E2M3FNUZ>());
    assert!(T::check::<E2M3FN>());

    assert!(T::check::<F6E3M2>());
    assert!(T::check::<F6E3M2FN>());
    assert!(T::check::<F6E3M2FNUZ>());
    assert!(T::check::<E3M2FN>());

    assert!(T::check::<F4E2M1>());
    assert!(T::check::<F4E2M1FN>());
    assert!(T::check::<F4E2M1FNUZ>());
    assert!(T::check::<E2M1FN>());

    test_8_bits(x);
}

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
fn test_arithmetic_matches_f64() {
    struct CheckArith;
    impl Check for CheckArith {
        fn check<T: Minifloat + Debug + Hash>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for_all::<T>(|x| {
                for_all::<T>(|y| {
                    let xf = x.to_f64();
                    let yf = y.to_f64();
                    same_mini(x + y, T::from_f64(xf + yf))
                        && same_mini(x - y, T::from_f64(xf - yf))
                        && same_mini(x * y, T::from_f64(xf * yf))
                        && same_mini(x / y, T::from_f64(xf / yf))
                })
            })
        }
    }
    test_most_8_bits(CheckArith);
}

#[test]
fn test_compound_assignment() {
    let mut x = F8E4M3FN::from_f64(1.5);
    x += F8E4M3FN::from_f64(0.5);
    assert_eq!(x.to_f64(), 2.0);
    x -= F8E4M3FN::from_f64(0.5);
    assert_eq!(x.to_f64(), 1.5);
    x *= F8E4M3FN::from_f64(2.0);
    assert_eq!(x.to_f64(), 3.0);
    x /= F8E4M3FN::from_f64(2.0);
    assert_eq!(x.to_f64(), 1.5);
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

/// Textbook value of a bit pattern, computed from the raw fields
///
/// The value computation deliberately shares no code with the crate: it is
/// the definition a human would read off the format specification.
fn oracle(bits: Mask, e_width: u32, m_width: u32, bias: i32, format: Format) -> f64 {
    let sign = bits >> (e_width + m_width) & 1;
    let exponent = bits >> m_width & bit_mask(e_width);
    let mantissa = bits & bit_mask(m_width);
    let sign = if sign == 1 { -1.0 } else { 1.0 };

    match format {
        Format::IEEE if exponent == bit_mask(e_width) => {
            return if mantissa == 0 { sign * f64::INFINITY } else { f64::NAN };
        }
        Format::FN if exponent == bit_mask(e_width) && mantissa == bit_mask(m_width) => {
            return f64::NAN;
        }
        Format::FNUZ if sign < 0.0 && exponent == 0 && mantissa == 0 => return f64::NAN,
        _ => {}
    }

    #[allow(clippy::cast_precision_loss)]
    let (significand, exponent) = if exponent == 0 {
        (mantissa as f64, 1 - bias)
    } else {
        (mantissa as f64 + f64::exp2(f64::from(m_width)), exponent as i32 - bias)
    };
    #[allow(clippy::cast_possible_wrap)]
    let value = scaled(significand, exponent - m_width as i32);
    sign * value
}

/// Check every bit pattern of `T` against [`oracle`]
///
/// The parameters restate the format independently of `T`'s declaration, so a
/// type declared with the wrong exponent bias or the wrong format fails here.
fn check_oracle<T: Minifloat>(e_width: u32, m_width: u32, bias: i32, format: Format)
where
    T::Bits: TryFrom<Mask>,
{
    assert_eq!((T::E, T::M, T::B), (e_width, m_width, bias));
    assert_eq!(T::FORMAT, format);
    assert_eq!(T::NAN.is_some(), format != Format::Finite);
    assert_eq!(T::INFINITY.is_some(), format == Format::IEEE);
    assert_eq!(T::HAS_NEG_ZERO, format != Format::FNUZ);

    for bits in 0..=bit_mask(T::BITWIDTH) {
        let expected = oracle(bits, e_width, m_width, bias, format);
        let actual = T::from_bits(narrow::<T>(bits)).to_f64();
        assert!(
            same_f64(actual, expected),
            "bits {bits:#x}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_predefined_types_match_their_specification() {
    check_oracle::<F4E2M1FN>(2, 1, 1, Format::Finite);
    check_oracle::<F6E2M3FN>(2, 3, 1, Format::Finite);
    check_oracle::<F6E3M2FN>(3, 2, 3, Format::Finite);
    check_oracle::<F8E3M4>(3, 4, 3, Format::IEEE);
    check_oracle::<F8E4M3>(4, 3, 7, Format::IEEE);
    check_oracle::<F8E4M3FN>(4, 3, 7, Format::FN);
    check_oracle::<F8E4M3FNUZ>(4, 3, 8, Format::FNUZ);
    check_oracle::<F8E4M3B11FNUZ>(4, 3, 11, Format::FNUZ);
    check_oracle::<F8E5M2>(5, 2, 15, Format::IEEE);
    check_oracle::<F8E5M2FNUZ>(5, 2, 16, Format::FNUZ);
    check_oracle::<F16>(5, 10, 15, Format::IEEE);
    check_oracle::<BF16>(8, 7, 127, Format::IEEE);
}

#[test]
fn test_predefined_maxima() {
    // OCP MX: the all-ones magnitude is a value, not a NaN.
    const _: () = assert!(!F4E2M1FN::HAS_NAN);
    const _: () = assert!(!F4E2M1FN::HAS_INF);
    assert_eq!(F4E2M1FN::MAX.to_f64(), 6.0);
    assert_eq!(F6E2M3FN::MAX.to_f64(), 7.5);
    assert_eq!(F6E3M2FN::MAX.to_f64(), 28.0);

    // The format is not the name suffix: `FN` still means all-ones-is-NaN.
    minifloat!(struct T(u8): 2, 1, FN);
    const _: () = assert!(T::HAS_NAN);
    assert_eq!(T::MAX.to_f64(), 4.0);

    // LLVM's `FNUZ` bias is 2^(E-1), not 2^(E-1) - 1.
    const _: () = assert!(F8E4M3FNUZ::B == 8);
    const _: () = assert!(F8E5M2FNUZ::B == 16);
    assert_eq!(F8E4M3FNUZ::MAX.to_f64(), 240.0);
    assert_eq!(F8E5M2FNUZ::MAX.to_f64(), 57344.0);
    assert_eq!(F8E4M3B11FNUZ::MAX.to_f64(), 30.0);

    // The table in README.md
    assert_eq!(F8E3M4::MAX.to_f64(), 15.5);
    assert_eq!(F8E4M3::MAX.to_f64(), 240.0);
    assert_eq!(F8E4M3FN::MAX.to_f64(), 448.0);
    assert_eq!(F8E5M2::MAX.to_f64(), 57344.0);
    assert_eq!(F16::MAX.to_f64(), 65504.0);
    assert_eq!(BF16::MAX.to_f64(), f64::exp2(128.0) * (1.0 - f64::exp2(-8.0)));
}

#[test]
fn test_max_10_exp() {
    // `MAX_EXP` already accounts for NaN eating the top row, so an `FN` type
    // with no mantissa bits needs no special case.
    minifloat!(struct E4M0FN(u8): 4, 0, 8, FN);
    const _: () = assert!(E4M0FN::MAX_EXP == 7);
    const _: () = assert!(E4M0FN::MAX_10_EXP == 1);
    assert_eq!(E4M0FN::MAX.to_f64(), 64.0);

    const _: () = assert!(F16::MAX_10_EXP == 4);
    const _: () = assert!(BF16::MAX_10_EXP == 38);
    const _: () = assert!(F8E4M3FN::MAX_10_EXP == 2);
    const _: () = assert!(F4E2M1FN::MAX_10_EXP == 0);
    const _: () = assert!(F6E3M2FN::MAX_10_EXP == 1);
}

/// Decompose a finite [`f64`] into an exact `(significand, exponent)` pair
///
/// The value is `significand` &times; 2<sup>`exponent`</sup> with no hidden
/// bits, so the pair can be compared and scaled without rounding.  The sign is
/// dropped; callers pass a magnitude.
fn decompose(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let field = (bits >> (f64::MANTISSA_DIGITS - 1)) as i32;
    let fraction = bits & bit_mask(f64::MANTISSA_DIGITS - 1);

    #[allow(clippy::cast_possible_wrap)]
    if field == 0 {
        (fraction, f64::MIN_EXP - f64::MANTISSA_DIGITS as i32)
    } else {
        (
            fraction | 1 << (f64::MANTISSA_DIGITS - 1),
            field + f64::MIN_EXP - 1 - f64::MANTISSA_DIGITS as i32,
        )
    }
}

/// Compare two non-negative `significand` &times; 2<sup>`exponent`</sup> pairs
///
/// The comparison is exact for any exponents, which the plain product is not:
/// both operands here routinely sit outside [`f64`]'s range.
fn cmp_scaled(a: (u64, i32), b: (u64, i32)) -> Ordering {
    if a.0 == 0 || b.0 == 0 {
        return a.0.cmp(&b.0);
    }
    #[allow(clippy::cast_possible_wrap)]
    let leading = |(significand, exponent): (u64, i32)| {
        exponent + (u64::BITS - 1 - significand.leading_zeros()) as i32
    };
    match leading(a).cmp(&leading(b)) {
        Ordering::Equal => {}
        decided => return decided,
    }
    // Equal leading-bit positions bound the exponent gap by the significand
    // width, so neither shift below can lose a bit.
    if a.1 >= b.1 {
        (u128::from(a.0) << (a.1 - b.1)).cmp(&u128::from(b.0))
    } else {
        u128::from(a.0).cmp(&(u128::from(b.0) << (b.1 - a.1)))
    }
}

/// Exact value of a magnitude code, as `significand` &times; 2<sup>`exponent`</sup>
///
/// This is the textbook decoding [`oracle`] uses, kept exact and extended one
/// code past the maximum so the overflow boundary has a candidate of its own.
fn code_value<T: Minifloat>(code: Mask) -> (u64, i32) {
    let field = code >> T::M;
    let fraction = code & bit_mask(T::M);

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    if field == 0 {
        (fraction, 1 - T::B - T::M as i32)
    } else {
        (fraction | 1 << T::M, field as i32 - T::B - T::M as i32)
    }
}

/// Magnitudes of the overflow result and of the maximum finite value
///
/// These restate the encoding rules instead of reading them off the type, so a
/// type whose `HUGE` or `MAX` drifts fails the checks that use them.
fn huge_and_max<T: Minifloat>() -> (Mask, Mask) {
    let abs_mask = bit_mask(T::E + T::M);
    let huge = match T::FORMAT {
        Format::IEEE => bit_mask(T::E) << T::M,
        Format::FN => abs_mask - 1,
        _ => abs_mask, // Finite, FNUZ
    };
    (huge, huge - Mask::from(T::FORMAT == Format::IEEE))
}

/// Correctly rounded encoding of `x`, derived from the format alone
///
/// The crate encodes by shifting exponent fields around; this one brackets `x`
/// between two neighbouring codes and picks the nearer, so the two share no
/// arithmetic.  Ties go to the even code, overflow to `HUGE`, and a NaN to the
/// format's NaN pattern (or to &plusmn;`MAX` where the format has none).
fn reference_encode<T: Minifloat>(x: f64) -> T
where
    T::Bits: TryFrom<Mask>,
{
    let sign_bit = Mask::from(x.is_sign_negative()) << (T::E + T::M);
    let (huge, max) = huge_and_max::<T>();

    if x.is_nan() {
        let magnitude = match T::FORMAT {
            Format::Finite => max,
            Format::IEEE => bit_mask(T::E + 1) << (T::M - 1),
            Format::FN => bit_mask(T::E + T::M),
            _ => 1 << (T::E + T::M), // FNUZ: the would-be negative zero
        };
        return T::from_bits(narrow::<T>(magnitude | sign_bit));
    }

    // Code values increase monotonically, so a binary search brackets `x`.
    let magnitude = decompose(x.abs());
    let mut lo = 0;
    let mut hi = max + 1;

    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cmp_scaled(code_value::<T>(mid), magnitude) == Ordering::Greater {
            hi = mid - 1;
        } else {
            lo = mid;
        }
    }

    let code = if lo > max {
        huge
    } else {
        let (significand, exponent) = code_value::<T>(lo);
        // Neighbouring codes differ by exactly one ULP of the lower one, so
        // their midpoint is `(2 * significand + 1) / 2` at the same scale.
        let up = match cmp_scaled(magnitude, (2 * significand + 1, exponent - 1)) {
            Ordering::Less => false,
            Ordering::Greater => true,
            Ordering::Equal => lo & 1 == 1,
        };
        (lo + Mask::from(up)).min(huge)
    };

    // Without a negative zero, signing a zero would spell NaN instead.
    let signed = T::HAS_NEG_ZERO || code != 0;
    T::from_bits(narrow::<T>(code | (Mask::from(signed) * sign_bit)))
}

/// Every input that changes which way `T` rounds
///
/// For each code this is the value itself, the midpoint to the next code up,
/// and the neighbours of that midpoint — the exactness, ties-to-even, and
/// nearest cases — plus the negation of each and the boundaries of [`f64`].
fn rounding_inputs<T: Minifloat>() -> Vec<f64> {
    let (_, max) = huge_and_max::<T>();
    let mut inputs = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF4_0000_0000_0BAD), // signalling NaN with a payload
        f64::from_bits(0xFFF8_0000_DEAD_BEEF), // negative NaN with a payload
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
    ];

    for code in 0..=max + 1 {
        let (significand, exponent) = code_value::<T>(code);
        #[allow(clippy::cast_precision_loss)]
        let value = scaled(significand as f64, exponent);
        #[allow(clippy::cast_precision_loss)]
        let midpoint = scaled((2 * significand + 1) as f64, exponent - 1);

        for x in [value, midpoint, midpoint.next_up(), midpoint.next_down()] {
            inputs.push(x);
            inputs.push(-x);
        }
    }
    inputs
}

/// Deterministic pseudo-random source
///
/// Knuth's MMIX multiplier over a fixed seed: no dependency, and a failure
/// always reproduces.  Only the high half of the state is handed out, since an
/// LCG's low bits cycle far too regularly.
struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_possible_truncation)]
        {
            (self.0 >> 32) as u32
        }
    }
}

/// Wide shapes reaching the conversion paths no 8-bit shape can
///
/// | shape | why |
/// |---|---|
/// | `11, 4` | exact in `f64` but not in `f32` |
/// | `12, 3` | exact in neither, default bias |
/// | `12, 3, 1000` | exact in neither, custom bias |
/// | `2, 13` | the widest mantissa a 16-bit shape can hold |
fn test_16_bits<T: Check>(_: T) {
    minifloat!(struct E11M4(u16): 11, 4);
    minifloat!(struct E12M3(u16): 12, 3);
    minifloat!(struct E12M3B1000(u16): 12, 3, (1000), IEEE);
    minifloat!(struct E12M3FN(u16): 12, 3, FN);
    minifloat!(struct E12M3FNUZ(u16): 12, 3, FNUZ);
    minifloat!(struct E12M3Finite(u16): 12, 3, Finite);
    minifloat!(struct E2M13(u16): 2, 13);

    assert!(T::check::<E11M4>());
    assert!(T::check::<E12M3>());
    assert!(T::check::<E12M3B1000>());
    assert!(T::check::<E12M3FN>());
    assert!(T::check::<E12M3FNUZ>());
    assert!(T::check::<E12M3Finite>());
    assert!(T::check::<E2M13>());

    assert!(T::check::<F16>());
    assert!(T::check::<BF16>());
}

#[test]
fn test_wide_shapes_match_their_specification() {
    minifloat!(struct E11M4(u16): 11, 4);
    minifloat!(struct E12M3(u16): 12, 3);
    minifloat!(struct E12M3B1000(u16): 12, 3, (1000), IEEE);
    minifloat!(struct E12M3FN(u16): 12, 3, FN);
    minifloat!(struct E12M3FNUZ(u16): 12, 3, FNUZ);
    minifloat!(struct E12M3Finite(u16): 12, 3, Finite);
    minifloat!(struct E2M13(u16): 2, 13);

    check_oracle::<E11M4>(11, 4, 1023, Format::IEEE);
    check_oracle::<E12M3>(12, 3, 2047, Format::IEEE);
    check_oracle::<E12M3B1000>(12, 3, 1000, Format::IEEE);
    check_oracle::<E12M3FN>(12, 3, 2047, Format::FN);
    check_oracle::<E12M3FNUZ>(12, 3, 2048, Format::FNUZ);
    check_oracle::<E12M3Finite>(12, 3, 2047, Format::Finite);
    check_oracle::<E2M13>(2, 13, 1, Format::IEEE);
}

#[test]
fn test_encode_correct_rounding() {
    struct CheckEncode;
    impl Check for CheckEncode {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for x in rounding_inputs::<T>() {
                let expected = reference_encode::<T>(x);
                let actual = T::from_f64(x);
                assert!(
                    actual.to_bits() == expected.to_bits(),
                    "from_f64({x:e}): got {actual:?}, expected {expected:?}"
                );

                // `from_f32` must agree wherever the input survives the trip
                // through `f32`.  NaNs survive as NaNs, payload aside.
                #[allow(clippy::cast_possible_truncation)]
                let narrow = x as f32;
                if same_f64(f64::from(narrow), x) {
                    let via_f32 = T::from_f32(narrow);
                    assert!(
                        via_f32.to_bits() == actual.to_bits(),
                        "from_f32({narrow:e}): got {via_f32:?}, expected {actual:?}"
                    );
                }
            }
            true
        }
    }
    test_most_8_bits(CheckEncode);
    test_16_bits(CheckEncode);
}

#[test]
fn test_encode_random_sweep() {
    struct CheckSweep;
    impl Check for CheckSweep {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            let mut lcg = Lcg::new(0x1234_5678_9ABC_DEF0);
            for _ in 0..1 << 14 {
                let x = f32::from_bits(lcg.next());
                let wide = f64::from(x);
                let expected = reference_encode::<T>(wide);
                let actual = T::from_f32(x);
                assert!(
                    actual.to_bits() == expected.to_bits(),
                    "from_f32({x:e}): got {actual:?}, expected {expected:?}"
                );
                assert!(
                    T::from_f64(wide).to_bits() == actual.to_bits(),
                    "from_f64({wide:e}) disagrees with from_f32({x:e})"
                );
            }
            true
        }
    }
    test_most_8_bits(CheckSweep);
    test_16_bits(CheckSweep);
}

#[test]
fn test_arithmetic_16bit_sampled() {
    struct CheckArith16;
    impl Check for CheckArith16 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            let mut lcg = Lcg::new(0x0FED_CBA9_8765_4321);
            let draw = |lcg: &mut Lcg| T::from_bits(narrow::<T>(Mask::from(lcg.next()) & bit_mask(T::BITWIDTH)));

            for _ in 0..1 << 13 {
                let x = draw(&mut lcg);
                let y = draw(&mut lcg);
                let (xf, yf) = (x.to_f64(), y.to_f64());

                // Operands whose `f64` image does not come back unchanged are
                // outside `f64`'s range, where it cannot referee the result.
                if !same_mini(T::from_f64(xf), x) || !same_mini(T::from_f64(yf), y) {
                    continue;
                }

                for (op, result) in [
                    ('+', (x + y, xf + yf)),
                    ('-', (x - y, xf - yf)),
                    ('*', (x * y, xf * yf)),
                    ('/', (x / y, xf / yf)),
                ] {
                    let (actual, exact) = result;
                    // A subnormal `f64` result has already lost bits of its own.
                    if exact != 0.0 && exact.abs() < f64::MIN_POSITIVE {
                        continue;
                    }
                    let expected = reference_encode::<T>(exact);
                    assert!(
                        same_mini(actual, expected),
                        "{x:?} {op} {y:?}: got {actual:?}, expected {expected:?}"
                    );
                }
            }
            true
        }
    }
    test_16_bits(CheckArith16);
}
