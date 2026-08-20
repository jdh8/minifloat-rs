// This file is part of the minifloat project.
//
// Copyright (C) 2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::encode::reference_encode;
use crate::support::*;
use minifloat::{Minifloat, F8E4M3FN};

use core::fmt::Debug;
use core::hash::Hash;

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
