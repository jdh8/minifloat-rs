// This file is part of the minifloat project.
//
// Copyright (C) 2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use minifloat::{
    minifloat, Minifloat, BF16, F16, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3, F8E4M3B11FNUZ,
    F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
};

use num_traits::AsPrimitive;

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
    fn check<T: Minifloat + Debug + Hash>() -> bool
    where
        Mask: AsPrimitive<T::Bits>;
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
            Mask: AsPrimitive<T::Bits>,
        {
            let fixed_point = if T::M == 0 { 2.0 } else { 3.0 };
            assert!(same_f32(T::from_f32(fixed_point).to_f32(), fixed_point));

            let fixed_point = f64::from(fixed_point);
            assert!(same_f64(T::from_f64(fixed_point).to_f64(), fixed_point));

            assert!(T::ZERO.to_bits() == 0.as_());
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
}

#[test]
fn test_neg() {
    struct CheckNeg;
    impl Check for CheckNeg {
        fn check<T: Minifloat + Debug>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
        {
            for_all::<T>(|x| x.to_bits() == (-(-x)).to_bits())
        }
    }
    test_most_8_bits(CheckNeg);
}

#[test]
fn test_partial_cmp() {
    struct CheckOrd;
    impl Check for CheckOrd {
        fn check<T: Minifloat + Debug>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
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
            Mask: AsPrimitive<T::Bits>,
        {
            for_all::<T>(|x| {
                u32::from(x.is_nan()) << FpCategory::Nan as u8
                    | u32::from(x.is_infinite()) << FpCategory::Infinite as u8
                    | u32::from(x.is_normal()) << FpCategory::Normal as u8
                    | u32::from(x.is_subnormal()) << FpCategory::Subnormal as u8
                    | u32::from(x == T::from_bits(0.as_())) << FpCategory::Zero as u8
                    == 1 << x.classify() as u8
            })
        }
    }
    test_most_8_bits(CheckClassify);
}

#[test]
fn test_to_f32() {
    struct CheckToF32;
    impl Check for CheckToF32 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
        {
            assert!(same_f32(T::ZERO.to_f32(), 0.0));
            assert!(same_f32(
                (-T::ZERO).to_f32(),
                if T::HAS_NEG_ZERO { -0.0 } else { 0.0 }
            ));
            for_all::<T>(|x| same_mini(T::from_f32(x.to_f32()), x))
        }
    }
    test_most_8_bits(CheckToF32);
}

#[test]
fn test_to_f64() {
    struct CheckToF64;
    impl Check for CheckToF64 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
        {
            assert!(same_f64(T::ZERO.to_f64(), 0.0));
            assert!(same_f64(
                (-T::ZERO).to_f64(),
                if T::HAS_NEG_ZERO { -0.0 } else { 0.0 }
            ));
            for_all::<T>(|x| same_mini(T::from_f64(x.to_f64()), x))
        }
    }
    test_most_8_bits(CheckToF64);
}

#[test]
fn test_to_floats() {
    struct CheckToFloats;
    impl Check for CheckToFloats {
        fn check<T: Minifloat + Debug>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
        {
            for_all::<T>(|x| same_f64(x.to_f32().into(), x.to_f64()))
        }
    }
    test_most_8_bits(CheckToFloats);
}

#[test]
fn test_arithmetic_matches_f64() {
    struct CheckArith;
    impl Check for CheckArith {
        fn check<T: Minifloat + Debug + Hash>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
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
    use core::cmp::Ordering;
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
        Mask: AsPrimitive<T::Bits>,
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
            Mask: AsPrimitive<T::Bits>,
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
                let reconstructed = f64::from(sign) * mantissa as f64 * libm::exp2(f64::from(exponent));
                same_f64(reconstructed, x.to_f64())
            })
        }
    }
    test_most_8_bits(CheckIntegerDecode);
}

#[test]
fn test_hash_consistent_with_eq() {
    struct CheckHash;
    impl Check for CheckHash {
        fn check<T: Minifloat + Debug + Hash>() -> bool
        where
            Mask: AsPrimitive<T::Bits>,
        {
            let state = std::collections::hash_map::RandomState::new();
            for_all::<T>(|x| {
                for_all::<T>(|y| x != y || state.hash_one(x) == state.hash_one(y))
            })
        }
    }
    test_most_8_bits(CheckHash);
}

/// Format of a minifloat, spelled out independently of the crate
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    /// Every bit pattern is a finite number
    Finite,
    /// Maximum exponent reserved for ±∞ and NaN
    Ieee,
    /// All-ones magnitude reserved for NaN
    Fn,
    /// &minus;0.0 encoding reserved for NaN
    Fnuz,
}

/// Textbook value of a bit pattern, computed from the raw fields
///
/// This deliberately shares no code with the crate: it is the definition a
/// human would read off the format specification.
fn oracle(bits: Mask, e_width: u32, m_width: u32, bias: i32, format: Format) -> f64 {
    let sign = bits >> (e_width + m_width) & 1;
    let exponent = bits >> m_width & bit_mask(e_width);
    let mantissa = bits & bit_mask(m_width);
    let sign = if sign == 1 { -1.0 } else { 1.0 };

    match format {
        Format::Ieee if exponent == bit_mask(e_width) => {
            return if mantissa == 0 { sign * f64::INFINITY } else { f64::NAN };
        }
        Format::Fn if exponent == bit_mask(e_width) && mantissa == bit_mask(m_width) => {
            return f64::NAN;
        }
        Format::Fnuz if sign < 0.0 && exponent == 0 && mantissa == 0 => return f64::NAN,
        _ => {}
    }

    let (significand, exponent) = if exponent == 0 {
        (mantissa as f64, 1 - bias)
    } else {
        (mantissa as f64 + libm::exp2(f64::from(m_width)), exponent as i32 - bias)
    };
    #[allow(clippy::cast_possible_wrap)]
    let scale = libm::exp2(f64::from(exponent - m_width as i32));
    sign * significand * scale
}

/// Check every bit pattern of `T` against [`oracle`]
///
/// The parameters restate the format independently of `T`'s declaration, so a
/// type declared with the wrong exponent bias or the wrong format fails here.
fn check_oracle<T: Minifloat>(e_width: u32, m_width: u32, bias: i32, format: Format)
where
    Mask: AsPrimitive<T::Bits>,
{
    assert_eq!((T::E, T::M, T::B), (e_width, m_width, bias));
    assert_eq!(T::NAN.is_some(), format != Format::Finite);
    assert_eq!(T::INFINITY.is_some(), format == Format::Ieee);
    assert_eq!(T::HAS_NEG_ZERO, format != Format::Fnuz);

    for bits in 0..=bit_mask(T::BITWIDTH) {
        let expected = oracle(bits, e_width, m_width, bias, format);
        let actual = T::from_bits(bits.as_()).to_f64();
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
    check_oracle::<F8E3M4>(3, 4, 3, Format::Ieee);
    check_oracle::<F8E4M3>(4, 3, 7, Format::Ieee);
    check_oracle::<F8E4M3FN>(4, 3, 7, Format::Fn);
    check_oracle::<F8E4M3FNUZ>(4, 3, 8, Format::Fnuz);
    check_oracle::<F8E4M3B11FNUZ>(4, 3, 11, Format::Fnuz);
    check_oracle::<F8E5M2>(5, 2, 15, Format::Ieee);
    check_oracle::<F8E5M2FNUZ>(5, 2, 16, Format::Fnuz);
    check_oracle::<F16>(5, 10, 15, Format::Ieee);
    check_oracle::<BF16>(8, 7, 127, Format::Ieee);
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
    assert_eq!(BF16::MAX.to_f64(), libm::exp2(128.0) * (1.0 - libm::exp2(-8.0)));
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
