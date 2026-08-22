// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::support::*;
use minifloat::{detail::decompose, Format, Minifloat};

use core::cmp::Ordering;
use core::fmt::Debug;

/// Compare two non-negative `significand` &times; 2<sup>`exponent`</sup> pairs
///
/// The comparison is exact for any exponents, which the plain product is not:
/// both operands here routinely sit outside [`f64`]'s range.  The significands
/// are [`u128`] because an exact sum of two operands needs more room than
/// either of them.
pub(crate) fn cmp_scaled(a: (u128, i32), b: (u128, i32)) -> Ordering {
    if a.0 == 0 || b.0 == 0 {
        return a.0.cmp(&b.0);
    }
    let leading = |(significand, exponent): (u128, i32)| {
        exponent + (u128::BITS - 1 - significand.leading_zeros()).cast_signed()
    };
    match leading(a).cmp(&leading(b)) {
        Ordering::Equal => {}
        decided => return decided,
    }
    // Equal leading-bit positions bound the exponent gap by the significand
    // width, so neither shift below can lose a bit.
    if a.1 >= b.1 {
        (a.0 << (a.1 - b.1)).cmp(&b.0)
    } else {
        a.0.cmp(&(b.0 << (b.1 - a.1)))
    }
}

/// Exact value of a magnitude code, as `significand` &times; 2<sup>`exponent`</sup>
///
/// This is the textbook decoding [`reference_round`] uses, kept exact and
/// extended one code past the maximum so the overflow boundary has a candidate
/// of its own.
pub(crate) fn code_value<T: Minifloat>(code: Mask) -> (u64, i32) {
    let field = code >> T::M;
    let fraction = code & bit_mask(T::M);

    #[allow(clippy::cast_possible_truncation)]
    if field == 0 {
        (fraction, 1 - T::B - T::M.cast_signed())
    } else {
        (fraction | 1 << T::M, field as i32 - T::B - T::M.cast_signed())
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

/// Correctly rounded encoding of a value reachable only by comparison
///
/// `cmp` reports how the exact value compares against a `significand` &times;
/// 2<sup>`exponent`</sup> pair, which is all rounding needs: the search
/// brackets the value between two neighbouring codes and the midpoint decides
/// which is nearer.  The value itself never has to be a number any float type
/// can hold, so an exact sum, product, or quotient can referee itself.
///
/// Ties go to the even code and overflow to `HUGE`.
pub(crate) fn reference_round<T: Minifloat>(
    negative: bool,
    cmp: impl Fn((u64, i32)) -> Ordering,
) -> T
where
    T::Bits: TryFrom<Mask>,
{
    let (huge, max) = huge_and_max::<T>();

    // Code values increase monotonically, so a binary search brackets the value.
    let mut lo = 0;
    let mut hi = max + 1;

    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cmp(code_value::<T>(mid)) == Ordering::Less {
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
        let up = match cmp((2 * significand + 1, exponent - 1)) {
            Ordering::Less => false,
            Ordering::Greater => true,
            Ordering::Equal => lo & 1 == 1,
        };
        (lo + Mask::from(up)).min(huge)
    };

    // Without a negative zero, signing a zero would spell NaN instead.
    let signed = T::HAS_NEG_ZERO || code != 0;
    let sign_bit = Mask::from(negative) << (T::E + T::M);
    T::from_bits(narrow::<T>(code | (Mask::from(signed) * sign_bit)))
}

/// Correctly rounded encoding of `x`, derived from the format alone
///
/// The crate encodes by shifting exponent fields around; this one brackets `x`
/// between two neighbouring codes and picks the nearer, so the *rounding*
/// shares nothing.  Reading the `f64` apart is not part of that: both sides
/// call `detail::decompose`, because an exact split of an `f64` has one right
/// answer, and the copy that used to live here was a copy rather than a second
/// opinion — it would have been wrong in step.  The integer oracle in
/// `arith.rs` is where independence is load-bearing, and it never touches an
/// `f64` at all.  Ties go to the even code, overflow to `HUGE`, and a NaN to
/// the format's NaN pattern (or to &plusmn;`MAX` where the format has none).
pub(crate) fn reference_encode<T: Minifloat>(x: f64) -> T
where
    T::Bits: TryFrom<Mask>,
{
    if x.is_nan() {
        let (_, max) = huge_and_max::<T>();
        let magnitude = match T::FORMAT {
            Format::Finite => max,
            Format::IEEE => bit_mask(T::E + 1) << (T::M - 1),
            Format::FN => bit_mask(T::E + T::M),
            _ => 1 << (T::E + T::M), // FNUZ: the would-be negative zero
        };
        let sign_bit = Mask::from(x.is_sign_negative()) << (T::E + T::M);
        return T::from_bits(narrow::<T>(magnitude | sign_bit));
    }

    let (significand, exponent) = decompose(x.abs());
    let magnitude = (u128::from(significand), exponent);
    reference_round(x.is_sign_negative(), |(s, e)| {
        cmp_scaled(magnitude, (u128::from(s), e))
    })
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
