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
