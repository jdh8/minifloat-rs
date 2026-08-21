// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::support::*;
use minifloat::{Minifloat, BF16, F16, F4E2M1FN, F8E3M4, F8E4M3FN, F8E4M3FNUZ};

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::{BuildHasher, Hash};
use core::num::{FpCategory, NonZeroUsize};

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

/// The order `total_cmp` owes us on one pair of encodings
///
/// Everything numeric comes from `partial_cmp`, so the key cannot referee
/// itself.  Only the two things a value comparison cannot express are stated in
/// encoding terms: the ±0 split, and where each format parks its NaN.
fn expected_total_cmp<T: Minifloat + Debug>(x: Mask, y: Mask) -> Ordering
where
    T::Bits: TryFrom<Mask>,
{
    let sign = 1 << (T::BITWIDTH - 1);
    let (xv, yv) = (T::from_bits(narrow::<T>(x)), T::from_bits(narrow::<T>(y)));

    // Sign-magnitude on the encoding.  This is the entire contract for a NaN,
    // which has no numeric position to be placed by: `IEEE` and `FN` keep the
    // sign bit, so their NaNs sit beyond ±`MAX` at either end, ordered by
    // payload; `FNUZ` spends the −0 encoding on NaN, so its NaN lands where −0
    // would, between the negatives and +0.
    let by_encoding = || match (x & sign, y & sign) {
        (0, 0) => x.cmp(&y),
        (0, _) => Ordering::Greater,
        (_, 0) => Ordering::Less,
        (_, _) => y.cmp(&x),
    };

    if xv.is_nan() || yv.is_nan() {
        return by_encoding();
    }

    match xv.partial_cmp(&yv) {
        // ±0 is the only numeric tie two distinct encodings can have, and the
        // total order breaks it by sign.
        Some(Ordering::Equal) => {
            assert!(
                x == y || (x | y) & (sign - 1) == 0,
                "distinct encodings {xv:?} and {yv:?} tie numerically without being ±0"
            );
            by_encoding()
        }
        Some(order) => order,
        None => unreachable!("a NaN operand would have taken the branch above"),
    }
}

/// Check `total_cmp` on one ordered pair of encodings
///
/// Trichotomy and antisymmetry come free: every caller sweeps ordered pairs, so
/// each pair is checked both ways against a single-valued expectation.
fn check_total_cmp<T: Minifloat + Debug>(x: Mask, y: Mask)
where
    T::Bits: TryFrom<Mask>,
{
    let (xv, yv) = (T::from_bits(narrow::<T>(x)), T::from_bits(narrow::<T>(y)));
    assert_eq!(
        xv.total_cmp(&yv),
        expected_total_cmp::<T>(x, y),
        "total_cmp({xv:?}, {yv:?})"
    );
}

#[test]
fn test_total_cmp() {
    struct CheckTotalCmp;
    impl Check for CheckTotalCmp {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            let mask = bit_mask(T::BITWIDTH);
            (0..=mask).all(|x| {
                (0..=mask).all(|y| {
                    check_total_cmp::<T>(x, y);
                    true
                })
            })
        }
    }
    test_most_8_bits(CheckTotalCmp);
}

/// Check both comparisons on every ordered pair of bit patterns
///
/// A 16-bit shape has 2<sup>32</sup> of them, so the left operand is striped
/// across the machine's cores, as `sweep_f32_pairs` does in `arith.rs`.  Every
/// stripe walks the whole right operand, so the sweep stays exhaustive however
/// many cores show up.
fn sweep_cmp_pairs<T: Minifloat + Debug>()
where
    T::Bits: TryFrom<Mask>,
{
    let stride = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let mask = bit_mask(T::BITWIDTH);

    std::thread::scope(|scope| {
        for offset in 0..stride as Mask {
            scope.spawn(move || {
                for x in (offset..=mask).step_by(stride) {
                    let xv = T::from_bits(narrow::<T>(x));

                    for y in 0..=mask {
                        let yv = T::from_bits(narrow::<T>(y));
                        check_total_cmp::<T>(x, y);
                        assert_eq!(
                            xv.partial_cmp(&yv),
                            xv.to_f32().partial_cmp(&yv.to_f32()),
                            "partial_cmp({xv:?}, {yv:?})"
                        );
                    }
                }
            });
        }
    });
}

/// `F16` and `BF16` are the 16-bit shapes worth 2<sup>32</sup> pairs apiece
///
/// The `E11M4` and `E12M3` shapes overrun `f32`'s exponent range, so a round
/// trip through it would referee the conversion instead of the comparison.
/// `E2M13` is exact in `f32` and would qualify &mdash; it is left out for the
/// pairs it would cost on a shape nothing ships, not for want of a referee.
/// `total_cmp` needs no referee and rides along.
#[test]
fn test_cmp_matches_f32_16bit() {
    sweep_cmp_pairs::<F16>();
    sweep_cmp_pairs::<BF16>();
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
                assert_eq!(
                    x.is_finite(),
                    !(x.is_nan() || x.is_infinite()),
                    "is_finite({x:?})"
                );
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
fn test_sign_and_abs() {
    struct CheckSign;
    impl Check for CheckSign {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            (0..=bit_mask(T::BITWIDTH)).all(|bits| {
                let x = T::from_bits(narrow::<T>(bits));
                let negative = bits >> (T::BITWIDTH - 1) != 0;
                assert_eq!(x.is_sign_negative(), negative, "{x:?}");
                assert_eq!(x.is_sign_positive(), !negative, "{x:?}");

                // `abs` clears the sign bit and touches nothing else.  Pin
                // the encoding, not the value: `same_f64` equates every NaN, so
                // a version that canonicalized an `IEEE` payload would pass a
                // value check while `Neg`, `Hash` and `total_cmp` all read that
                // payload as part of the number.
                let abs = x.abs();
                let expected = if !T::HAS_NEG_ZERO && x.is_nan() {
                    // Here NaN *is* the sign bit, so clearing it would turn the
                    // NaN into a zero.
                    x.to_bits()
                } else {
                    narrow::<T>(bits & bit_mask(T::E + T::M))
                };
                assert!(abs.to_bits() == expected, "abs({x:?}) = {abs:?}");
                true
            })
        }
    }
    test_most_8_bits(CheckSign);
    test_16_bits(CheckSign);
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

/// The `const` helpers agree with the operators over every pair
///
/// [`PartialEq`] and [`PartialOrd`] forward to them today, so this sweep is a
/// tautology — which is the point: it is what fails the day one of the two
/// grows a body of its own.  The macro body is shape-generic, so one shape per
/// format is the whole surface.
#[test]
fn test_const_cmp_equivalence() {
    macro_rules! sweep_shapes {
        ($($shape:ident)*) => {$({
            let mask = u8::MAX >> (u8::BITS - $shape::BITWIDTH);

            for x in 0..=mask {
                for y in 0..=mask {
                    let (x, y) = ($shape::from_bits(x), $shape::from_bits(y));
                    assert_eq!(x.const_eq(y), x == y, "{x:?} == {y:?}");
                    assert_eq!(
                        x.const_partial_cmp(y),
                        x.partial_cmp(&y),
                        "partial_cmp({x:?}, {y:?})"
                    );
                }
            }
        })*};
    }
    // One shape per format: `IEEE`, `FN`, `FNUZ`, and the `Finite` that the MX
    // types carry despite their `FN` names.
    sweep_shapes! { F8E3M4 F8E4M3FN F8E4M3FNUZ F4E2M1FN }
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
