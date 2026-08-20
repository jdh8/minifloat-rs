// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::encode::{cmp_scaled, code_value, reference_encode, reference_round};
use crate::support::*;
use minifloat::{minifloat, Minifloat, BF16, F16, F8E4M3FN};

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::Hash;
use core::num::NonZeroUsize;

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
                    for (op, actual, exact) in [
                        ('+', x + y, xf + yf),
                        ('-', x - y, xf - yf),
                        ('*', x * y, xf * yf),
                        ('/', x / y, xf / yf),
                    ] {
                        let expected = T::from_f64(exact);
                        // A format without a NaN saturates one to ±`MAX`.
                        // Which sign it lands on is the host's default NaN
                        // talking, not a rule, so only the magnitude binds.
                        let (actual, expected) = if exact.is_nan() && T::NAN.is_none() {
                            (actual.abs(), expected.abs())
                        } else {
                            (actual, expected)
                        };
                        assert!(
                            same_mini(actual, expected),
                            "{x:?} {op} {y:?}: got {actual:?}, expected {expected:?}"
                        );
                    }
                    true
                })
            })
        }
    }
    test_most_8_bits(CheckArith);
}

/// Check all four operators against a round trip through [`f32`] on one pair
///
/// Widening both operands, letting the FPU work, and rounding the result back
/// is the alternative `benches/arith.rs` measures against.  It has to land on
/// the same value, or the benchmark would be timing two different answers.
fn check_f32_pair<T: Minifloat + Debug>(x: T, y: T) {
    let (xf, yf) = (x.to_f32(), y.to_f32());

    for (op, actual, exact) in [
        ('+', x + y, xf + yf),
        ('-', x - y, xf - yf),
        ('*', x * y, xf * yf),
        ('/', x / y, xf / yf),
    ] {
        let expected = T::from_f32(exact);
        // A format without a NaN saturates one to ±`MAX`.  Which sign it lands
        // on is the host's default NaN talking, not a rule, so only the
        // magnitude binds.
        let (actual, expected) = if exact.is_nan() && T::NAN.is_none() {
            (actual.abs(), expected.abs())
        } else {
            (actual, expected)
        };
        assert!(
            same_mini(actual, expected),
            "{x:?} {op} {y:?}: got {actual:?}, expected {expected:?}"
        );
    }
}

/// Every shape up to 8 bits clears the bar for the `f32` route
///
/// Exact operands, and 2<var>p</var> + 2 digits left to round in: no shape
/// this narrow can fail either, so the sweep needs no filter.
#[test]
fn test_arithmetic_matches_f32() {
    struct CheckArith32;
    impl Check for CheckArith32 {
        fn check<T: Minifloat + Debug + Hash>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for_all::<T>(|x| {
                for_all::<T>(|y| {
                    check_f32_pair(x, y);
                    true
                })
            })
        }
    }
    test_most_8_bits(CheckArith32);
}

/// Run [`check_f32_pair`] on every ordered pair of bit patterns
///
/// A 16-bit shape has 2<sup>32</sup> of them, so the left operand is striped
/// across the machine's cores.  Every stripe walks the whole right operand, so
/// the sweep stays exhaustive however many cores show up.
fn sweep_f32_pairs<T: Minifloat + Debug>()
where
    T::Bits: TryFrom<Mask>,
{
    let stride = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let mask = bit_mask(T::BITWIDTH);

    std::thread::scope(|scope| {
        for offset in 0..stride as Mask {
            scope.spawn(move || {
                for x in (offset..=mask).step_by(stride) {
                    let x = T::from_bits(narrow::<T>(x));

                    for y in 0..=mask {
                        check_f32_pair(x, T::from_bits(narrow::<T>(y)));
                    }
                }
            });
        }
    });
}

/// `F16` and `BF16` are the only 16-bit shapes the `f32` route can serve
///
/// `E11M4` and `E12M3` are not exactly representable in `f32` to begin with,
/// and `E2M13` is, yet a product of two of its significands is 28 digits wide
/// &mdash; too many to round through 24.  `benches/arith.rs` sends those
/// through `f64`, or skips them.
#[test]
fn test_arithmetic_matches_f32_16bit() {
    sweep_f32_pairs::<F16>();
    sweep_f32_pairs::<BF16>();
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
fn test_arithmetic_16bit_sampled() {
    struct CheckArith16;
    impl Check for CheckArith16 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            // `f64` cannot referee a shape it cannot even hold; the exact
            // oracle in `test_arithmetic_correctly_rounded_16bit` does.
            if !T::HAS_EXACT_F64_CONVERSION {
                return true;
            }
            let mut lcg = Lcg::new(0x0FED_CBA9_8765_4321);
            let draw = |lcg: &mut Lcg| T::from_bits(narrow::<T>(Mask::from(lcg.next()) & bit_mask(T::BITWIDTH)));

            for _ in 0..1 << 13 {
                let x = draw(&mut lcg);
                let y = draw(&mut lcg);
                let (xf, yf) = (x.to_f64(), y.to_f64());

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

/// An exact arithmetic result, as a signed ratio
///
/// The value is &plusmn;`numerator` / `denominator`, where `numerator` is an
/// exact `significand` &times; 2<sup>`exponent`</sup> pair.  Only division ever
/// needs a denominator, but carrying one through every operation lets a single
/// comparison referee all four.
struct Exact {
    negative: bool,
    numerator: (u128, i32),
    denominator: u64,
}

/// Compare an exact result against `significand` &times; 2<sup>`exponent`</sup>
///
/// Cross-multiplying by the denominator keeps the comparison in integers, so
/// nothing here rounds.
fn cmp_exact(exact: &Exact, (significand, exponent): (u64, i32)) -> Ordering {
    let scaled = u128::from(significand) * u128::from(exact.denominator);
    cmp_scaled(exact.numerator, (scaled, exponent))
}

/// Widest exponent gap an aligned sum spans
///
/// Two addends further apart than this cannot both matter.  The smaller one is
/// then more than 49 binades below the nearest midpoint of the larger, so a
/// single unit in the last place stands in for it: still nonzero, so it breaks
/// an exact tie the same way, yet far too small to move any other comparison.
/// Capping the gap is what keeps the aligned sum inside an [`i128`].
const ALIGN_CAP: i32 = 64;

/// Exact sum of two signed magnitudes
fn exact_sum((xn, (xs, xe)): (bool, (u64, i32)), (yn, (ys, ye)): (bool, (u64, i32))) -> Exact {
    let base = xe.min(ye).max(xe.max(ye) - ALIGN_CAP);
    let scale = |negative: bool, significand: u64, exponent: i32| {
        let magnitude = if significand == 0 {
            0
        } else if exponent >= base {
            i128::from(significand) << (exponent - base)
        } else {
            1 // The sticky stand-in for a far-away addend
        };
        if negative {
            -magnitude
        } else {
            magnitude
        }
    };
    let sum = scale(xn, xs, xe) + scale(yn, ys, ye);

    Exact {
        // Cancellation yields +0 unless both addends were negative.
        negative: if sum == 0 { xn && yn } else { sum < 0 },
        numerator: (sum.unsigned_abs(), base),
        denominator: 1,
    }
}

/// Sign and exact magnitude of a bit pattern
fn operand<T: Minifloat>(bits: Mask) -> (bool, (u64, i32)) {
    (
        bits >> (T::E + T::M) != 0,
        code_value::<T>(bits & bit_mask(T::E + T::M)),
    )
}

/// Check all four operators on one pair of bit patterns
///
/// Both operands and the exact result stay in integers, so nothing here
/// inherits a rounding the implementation made.
fn check_pair<T: Minifloat + Debug>(x_bits: Mask, y_bits: Mask)
where
    T::Bits: TryFrom<Mask>,
{
    let x = T::from_bits(narrow::<T>(x_bits));
    let y = T::from_bits(narrow::<T>(y_bits));

    // An infinity or a NaN has no exact magnitude to compare against, and
    // `test_arithmetic_matches_f64` already pins what the operators do with
    // them.
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    let (xn, (xs, xe)) = operand::<T>(x_bits);
    let (yn, (ys, ye)) = operand::<T>(y_bits);

    let product = Exact {
        negative: xn != yn,
        numerator: (u128::from(xs) * u128::from(ys), xe + ye),
        denominator: 1,
    };
    if ys == 0 {
        // Zero over zero is invalid, and a format without a NaN saturates that
        // to `MAX`; anything else over zero overflows past every exponent.
        let expected = if xs == 0 {
            T::NAN.unwrap_or(T::MAX)
        } else if xn != yn {
            -T::HUGE
        } else {
            T::HUGE
        };
        let actual = x / y;
        assert!(
            same_mini(actual, expected),
            "{x:?} / {y:?}: got {actual:?}, expected {expected:?}"
        );
    }
    let quotient = (ys != 0).then(|| Exact {
        negative: xn != yn,
        numerator: (u128::from(xs), xe - ye),
        denominator: ys,
    });

    for (op, actual, exact) in [
        ('+', x + y, Some(exact_sum((xn, (xs, xe)), (yn, (ys, ye))))),
        ('-', x - y, Some(exact_sum((xn, (xs, xe)), (!yn, (ys, ye))))),
        ('*', x * y, Some(product)),
        ('/', x / y, quotient),
    ] {
        let Some(exact) = exact else { continue };
        let expected =
            reference_round::<T>(exact.negative, |candidate| cmp_exact(&exact, candidate));
        assert!(
            same_mini(actual, expected),
            "{x:?} {op} {y:?}: got {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn test_arithmetic_correctly_rounded() {
    struct CheckExact;
    impl Check for CheckExact {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            for x in 0..=bit_mask(T::BITWIDTH) {
                for y in 0..=bit_mask(T::BITWIDTH) {
                    check_pair::<T>(x, y);
                }
            }
            true
        }
    }
    test_most_8_bits(CheckExact);
}

#[test]
fn test_arithmetic_correctly_rounded_16bit() {
    struct CheckExact16;
    impl Check for CheckExact16 {
        fn check<T: Minifloat + Debug>() -> bool
        where
            T::Bits: TryFrom<Mask>,
        {
            let mut lcg = Lcg::new(0x1357_9BDF_0246_8ACE);
            let mut draw = || Mask::from(lcg.next()) & bit_mask(T::BITWIDTH);

            for _ in 0..1 << 16 {
                check_pair::<T>(draw(), draw());
            }
            true
        }
    }
    test_16_bits(CheckExact16);
}

#[test]
fn test_arithmetic_past_f64() {
    minifloat!(struct E12M3(u16): 12, 3);

    // Every code here is 1 × 2^k, whose exponent field is `k` + `B`.
    let power = |k: i32| E12M3::from_bits(((k + E12M3::B) << E12M3::M) as u16);
    let cases = [
        (power(-1000) * power(-1000), power(-2000)), // `f64` flushes this to zero
        (power(1000) * power(1000), power(2000)),    // and this to infinity
        (power(-2000) / power(-1000), power(-1000)),
        (power(2000) - power(1999), power(1999)),
    ];

    for (actual, expected) in cases {
        assert!(
            same_mini(actual, expected),
            "got {actual:?}, expected {expected:?}"
        );
    }
}
