// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

use minifloat::{
    minifloat, Minifloat, BF16, F16, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
    F8E4M3B11FNUZ, F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
};

use core::fmt::Debug;
use core::hash::Hash;

/// Bitmask returned by [`bit_mask`]
///
/// This type must be an unsigned integer.
pub(crate) type Mask = u64;

/// Create a bitmask of the given width
pub(crate) const fn bit_mask(width: u32) -> Mask {
    assert!(width <= Mask::BITS);

    if width == 0 {
        0
    } else {
        !0 >> (Mask::BITS - width)
    }
}

/// Narrow a bit pattern to a minifloat's storage
///
/// A minifloat is at most 16 bits wide, so the conversion always succeeds.
pub(crate) fn narrow<T: Minifloat>(bits: Mask) -> T::Bits
where
    T::Bits: TryFrom<Mask>,
{
    match bits.try_into() {
        Ok(narrowed) => narrowed,
        Err(_) => unreachable!("a minifloat is narrower than its storage"),
    }
}

/// Test floating-point identity like Object.is in JavaScript
///
/// This is necessary because NaN != NaN in C++.  We also want to differentiate
/// -0 from +0.  Using this functor, NaNs are considered identical to each
/// other, while +0 and -0 are considered different.
pub(crate) const fn same_f32(x: f32, y: f32) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

/// Test floating-point identity like Object.is in JavaScript
///
/// See also [`same_f32`].
pub(crate) const fn same_f64(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

/// <var>significand</var> &times; 2<sup><var>exponent</var></sup>, correctly rounded
///
/// Splitting the scale keeps either factor within [`f64`]'s exponent range, so
/// the first product is exact and the second rounds at most once.  A single
/// `exp2` would flush to zero or to infinity long before the product does.
pub(crate) fn scaled(significand: f64, exponent: i32) -> f64 {
    let head = exponent.clamp(f64::MIN_EXP - 1, f64::MAX_EXP - 1);
    significand * f64::exp2(f64::from(head)) * f64::exp2(f64::from(exponent - head))
}

/// Test floating-point identity like Object.is in JavaScript
///
/// See also [`same_f32`].
pub(crate) fn same_mini<T: Minifloat>(x: T, y: T) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

/// Iterate over all representations of a minifloat type
pub(crate) fn for_all<T: Minifloat>(f: impl Fn(T) -> bool) -> bool
where
    T::Bits: TryFrom<Mask>,
{
    (0..=bit_mask(T::BITWIDTH)).all(|bits| f(T::from_bits(narrow::<T>(bits))))
}

/// Wrapper trait for checking properties of minifloats
///
/// This trait helps building generic test infrastructure.  Opposed to generic
/// functions, traits can work as parameters.
pub(crate) trait Check {
    /// Check properties of a minifloat type
    fn check<T: Minifloat + Debug + Hash>() -> bool
    where
        T::Bits: TryFrom<Mask>;
}

/// Every shape in the roster narrower than 16 bits
///
/// The 4- and 6-bit shapes first, then the 8-bit ones.  A caller that wants
/// only the wide shapes calls [`test_16_bits`] instead; every caller that
/// wants the narrow ones wants all of them.
pub(crate) fn test_most_8_bits<T: Check>(_: T) {
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

/// Wide shapes reaching the conversion paths no 8-bit shape can
///
/// | shape | why |
/// |---|---|
/// | `11, 4` | exact in `f64` but not in `f32` |
/// | `12, 3` | exact in neither, default bias |
/// | `12, 3, 1000` | exact in neither, custom bias |
/// | `2, 13` | the widest mantissa a 16-bit shape can hold |
pub(crate) fn test_16_bits<T: Check>(_: T) {
    minifloat!(struct E11M4(u16): 11, 4);
    minifloat!(struct E12M3(u16): 12, 3);
    minifloat!(struct E12M3B1000(u16): 12, 3, (1000), IEEE);
    minifloat!(struct E12M3FN(u16): 12, 3, FN);
    minifloat!(struct E12M3FNUZ(u16): 12, 3, FNUZ);
    minifloat!(struct E12M3Finite(u16): 12, 3, Finite);
    minifloat!(struct E2M13(u16): 2, 13);

    assert!(T::check::<E11M4>());
    assert!(T::check::<E12M3>());
    assert!(T::check::<E12M3B1000>());
    assert!(T::check::<E12M3FN>());
    assert!(T::check::<E12M3FNUZ>());
    assert!(T::check::<E12M3Finite>());
    assert!(T::check::<E2M13>());

    assert!(T::check::<F16>());
    assert!(T::check::<BF16>());
}

/// Deterministic pseudo-random source
///
/// Knuth's MMIX multiplier over a fixed seed: no dependency, and a failure
/// always reproduces.  Only the high half of the state is handed out, since an
/// LCG's low bits cycle far too regularly.
pub(crate) struct Lcg(u64);

impl Lcg {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}
