// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
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
#[inline]
pub const fn exp2i(x: i32) -> f64 {
    f64::from_bits(match 0x3FF + x {
        0x800.. => 0x7FF << 52,
        s @ 1..=0x7FF => (s.cast_unsigned() as u64) << 52,
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
#[inline]
pub const fn decompose(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let field = (bits >> (f64::MANTISSA_DIGITS - 1)) as i32;
    let fraction = bits & ((1 << (f64::MANTISSA_DIGITS - 1)) - 1);

    if field == 0 {
        (fraction, f64::MIN_EXP - f64::MANTISSA_DIGITS.cast_signed())
    } else {
        (
            fraction | 1 << (f64::MANTISSA_DIGITS - 1),
            field + f64::MIN_EXP - 1 - f64::MANTISSA_DIGITS.cast_signed(),
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
/// The caller is responsible for the quotient fitting in an [`i64`]: every
/// significand handed in here is at most 2<sup>63</sup>, so a shift that only
/// ever narrows it has room to spare.
#[must_use]
#[inline]
pub const fn round_to_scale(significand: u64, exponent: i32, target: i32) -> i64 {
    let shift = target - exponent;

    if shift <= 0 {
        return (significand << -shift).cast_signed();
    }
    if shift >= u64::BITS.cast_signed() {
        return 0;
    }
    let dropped = significand & ((1 << shift) - 1);
    let kept = significand >> shift;
    let half = 1 << (shift - 1);
    (kept + (dropped > half || dropped == half && kept & 1 != 0) as u64).cast_signed()
}

/// log<sub>2</sub>(1 &minus; 2<sup>&minus;`p`</sup>) indexed by precision `p`
///
/// Adding this to a power of two gives the base-2 logarithm of the largest
/// value with that precision.  Index 0 is unreachable — the precision of a
/// maximum finite value is always at least 1 — and its entry is a leftover.
#[allow(clippy::excessive_precision)]
pub const LOG2_SIGNIFICAND: [f64; 32] = [
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
    -2.201_394_726_395_550_20e-5,
    -1.100_693_164_338_518_64e-5,
    -5.503_455_324_624_539_28e-6,
    -2.751_725_038_055_266_97e-6,
    -1.375_861_862_964_634_16e-6,
    -6.879_307_674_667_236_69e-7,
    -3.439_653_427_294_830_34e-7,
    -1.719_826_611_137_742_61e-7,
    -8.599_132_799_414_562_18e-8,
    -4.299_566_335_638_747_19e-8,
    -2.149_783_151_802_240_60e-8,
    -1.074_891_571_896_837_11e-8,
    -5.374_457_849_473_477_65e-9,
    -2.687_228_922_234_061_86e-9,
    -1.343_614_460_491_361_69e-9,
    -6.718_072_300_892_635_35e-10,
];

/// Widest exponent gap an aligned sum spans
///
/// An addend further below the other than this cannot move it at all: the
/// other is a representable value, and anything under half its ULP rounds
/// straight back to it.  That is the lower bound, <var>p</var> + 1, where
/// <var>p</var> &le; 30 is the widest significand a 32-bit shape holds.
///
/// Dropping such an addend is also what keeps the aligned sum inside an
/// [`i64`]: the sum stays under 2<sup><var>p</var>+CAP</sup> once the cap is at
/// least <var>p</var>, so the cap is at most 63 &minus; <var>p</var>.  The two
/// bounds leave [31, 33] at <var>p</var> = 30, and 32 sits inside it.
const ALIGN_CAP: i32 = 32;

/// One addend at the common exponent, signed
///
/// An addend below that exponent is one [`ALIGN_CAP`] has already ruled out.
#[inline]
const fn align(negative: bool, significand: u64, exponent: i32, base: i32) -> i64 {
    let magnitude = if exponent >= base {
        (significand << (exponent - base)).cast_signed()
    } else {
        0
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// Sum of two signed magnitudes, exact enough to round
///
/// Each operand is `sign` &times; `significand` &times;
/// 2<sup>`exponent`</sup>, and so is the result.  The caller is responsible
/// for the significands fitting in 30 bits, which every minifloat does.
#[must_use]
#[inline]
pub const fn add_parts(
    negative: bool,
    significand: u64,
    exponent: i32,
    rhs_negative: bool,
    rhs_significand: u64,
    rhs_exponent: i32,
) -> (bool, u64, i32) {
    let top = if exponent > rhs_exponent { exponent } else { rhs_exponent };
    let bottom = if exponent < rhs_exponent { exponent } else { rhs_exponent };
    let base = if top - bottom > ALIGN_CAP { top - ALIGN_CAP } else { bottom };

    let sum = align(negative, significand, exponent, base)
        + align(rhs_negative, rhs_significand, rhs_exponent, base);

    (
        // Cancellation yields +0 unless both addends were negative.
        if sum == 0 { negative && rhs_negative } else { sum < 0 },
        sum.unsigned_abs(),
        base,
    )
}

/// Quotient of two magnitudes, exact enough to round
///
/// The remainder collapses into the lowest bit of the quotient, the sticky bit
/// every divider keeps.  What has to be shown is that enough quotient bits sit
/// above it: rounding to <var>p</var> digits needs <var>p</var> + 2.
///
/// A fixed shift cannot deliver them.  A subnormal dividend &mdash; a
/// significand as small as 1 &mdash; over a full-width divisor leaves only
/// 63 &minus; 2<var>p</var> bits, which falls short once <var>p</var> &gt; 20.
/// Normalising the dividend to bit 62 instead leaves at least 63 &minus;
/// <var>p</var> whatever it was, and keeps the quotient under 2<sup>63</sup>
/// because the divisor is at least 1.
///
/// The caller is responsible for a nonzero divisor, for both significands
/// fitting in 30 bits &mdash; bit 63 of the dividend must be clear or the
/// shift below underflows &mdash; and for reading the sticky bit as sticky.
#[must_use]
#[inline]
pub const fn div_parts(
    significand: u64,
    exponent: i32,
    rhs_significand: u64,
    rhs_exponent: i32,
) -> (u64, i32) {
    // A zero dividend leaves a shift of 63 and a quotient of zero, which is
    // the right answer for it.
    let shift = significand.leading_zeros() - 1;
    let numerator = significand << shift;
    let quotient = numerator / rhs_significand;
    let remainder = numerator % rhs_significand;

    (
        quotient | (remainder != 0) as u64,
        exponent - rhs_exponent - shift.cast_signed(),
    )
}
