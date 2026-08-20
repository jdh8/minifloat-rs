// This file is part of the minifloat project.
//
// Copyright (C) 2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Implementation details
//!
//! This module is not considered part of the public API.  Symbols here are
//! public for macros to work.  They are not meant to be used directly.
#![doc(hidden)]

/// Fast 2<sup>`x`</sup> with bit manipulation
#[must_use]
pub const fn exp2i(x: i32) -> f64 {
    f64::from_bits(match 0x3FF + x {
        0x800.. => 0x7FF << 52,
        #[allow(clippy::cast_sign_loss)]
        s @ 1..=0x7FF => (s as u64) << 52,
        s @ -51..=0 => 1 << (51 + s),
        _ => 0,
    })
}

/// Decompose a [`f64`] into an exact `(significand, exponent)` pair
///
/// The value is <var>significand</var> &times;
/// 2<sup><var>exponent</var></sup> with no hidden bits, subnormal inputs
/// included.  The sign is dropped, so callers pass a magnitude.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub const fn decompose(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let field = (bits >> (f64::MANTISSA_DIGITS - 1)) as i32;
    let fraction = bits & ((1 << (f64::MANTISSA_DIGITS - 1)) - 1);

    if field == 0 {
        (fraction, f64::MIN_EXP - f64::MANTISSA_DIGITS as i32)
    } else {
        (
            fraction | 1 << (f64::MANTISSA_DIGITS - 1),
            field + f64::MIN_EXP - 1 - f64::MANTISSA_DIGITS as i32,
        )
    }
}

/// Round <var>significand</var> &times; 2<sup><var>exponent</var></sup> to a
/// multiple of 2<sup><var>target</var></sup>
///
/// Ties go to even, which is what IEEE 754 rounds to by default.  Working on
/// an integer significand keeps this exact for exponents far outside the range
/// of any hardware float.
///
/// The caller is responsible for the quotient fitting in an [`i64`]; a
/// minifloat code is at most 16 bits wide, so every call here has room to
/// spare.
#[must_use]
pub const fn round_to_scale(significand: u64, exponent: i32, target: i32) -> i64 {
    let shift = target - exponent;

    if shift <= 0 {
        return (significand << -shift) as i64;
    }
    if shift >= u64::BITS as i32 {
        return 0;
    }
    let dropped = significand & ((1 << shift) - 1);
    let kept = significand >> shift;
    let half = 1 << (shift - 1);
    (kept + (dropped > half || dropped == half && kept & 1 != 0) as u64) as i64
}

/// log<sub>2</sub>(1 &minus; 2<sup>&minus;`p`</sup>) indexed by precision `p`
///
/// Adding this to a power of two gives the base-2 logarithm of the largest
/// value with that precision.  Index 0 is unreachable — the precision of a
/// maximum finite value is always at least 1 — and its entry is a leftover.
#[allow(clippy::excessive_precision)]
pub const LOG2_SIGNIFICAND: [f64; 16] = [
    -2.0,
    -1.0,
    -4.150_374_992_788_438_13e-1,
    -1.926_450_779_423_958_81e-1,
    -9.310_940_439_148_146_51e-2,
    -4.580_368_961_312_478_86e-2,
    -2.272_007_650_008_352_89e-2,
    -1.131_531_322_783_414_61e-2,
    -5.646_563_141_142_062_72e-3,
    -2.820_519_062_378_662_63e-3,
    -1.409_570_254_671_353_63e-3,
    -7.046_129_765_893_727_06e-4,
    -3.522_634_716_290_213_85e-4,
    -1.761_209_842_740_240_62e-4,
    -8.805_780_458_002_638_34e-5,
    -4.402_823_044_177_721_15e-5,
];
