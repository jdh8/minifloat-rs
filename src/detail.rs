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

/// Round to the nearest representable value with `M` explicit bits of precision
#[must_use]
pub const fn round_f32_to_precision<const M: u32>(x: f32) -> f32 {
    let x = x.to_bits();
    let shift = f32::MANTISSA_DIGITS - 1 - M;
    let ulp = 1 << shift;
    let bias = (ulp >> 1) - (!(x >> shift) & 1);
    f32::from_bits((x + bias) & !(ulp - 1))
}

/// Round to the nearest representable value with `M` explicit bits of precision
#[must_use]
pub const fn round_f64_to_precision<const M: u32>(x: f64) -> f64 {
    let x = x.to_bits();
    let shift = f64::MANTISSA_DIGITS - 1 - M;
    let ulp = 1 << shift;
    let bias = (ulp >> 1) - (!(x >> shift) & 1);
    f64::from_bits((x + bias) & !(ulp - 1))
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
