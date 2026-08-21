// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::support::*;
use minifloat::{
    minifloat, Format, Minifloat, BF16, F16, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
    F8E4M3B11FNUZ, F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
};

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
    let value = scaled(significand, exponent - m_width.cast_signed());
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
