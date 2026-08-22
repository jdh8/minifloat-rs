// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The operand source both bench targets draw from
//!
//! Sharing it is what makes the two comparable: an A/B build differs in the
//! crate under test, never in how the operands were drawn.

use minifloat::Minifloat;

/// Operand pairs one measurement runs over
///
/// Small enough to stay in L1, so the loop times the crate and not memory.
pub(crate) const PAIRS: usize = 1 << 10;

/// Deterministic pseudo-random source
///
/// Knuth's MMIX multiplier over a fixed seed: no dependency, and every run
/// draws the same operands.  Only the high half of the state is handed out,
/// since an LCG's low bits cycle far too regularly.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}

/// Draw [`PAIRS`] operand pairs, one bit pattern at a time
///
/// Drawing raw codes gives NaNs, infinities and subnormals the density they
/// have in the format, which is the mix an operator or a predicate has to
/// survive.  Both routes of a comparison then run over the very same pairs; a
/// unary predicate reads the left operand of each.
pub(crate) fn pairs<T: Minifloat>() -> Vec<(T, T)>
where
    T::Bits: TryFrom<u32>,
{
    let mut lcg = Lcg(0x0FED_CBA9_8765_4321);
    let mask = u32::MAX >> (u32::BITS - T::BITWIDTH);
    let mut draw = || match (lcg.next() & mask).try_into() {
        Ok(bits) => T::from_bits(bits),
        Err(_) => unreachable!("a minifloat is narrower than its storage"),
    };
    (0..PAIRS).map(|_| (draw(), draw())).collect()
}
