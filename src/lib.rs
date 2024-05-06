// This file is part of the minifloat project.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#![no_std]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use core::marker::ConstParamTy;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, ConstParamTy, PartialEq, Eq)]
pub enum NanStyle {
    IEEE,
    FN,
    FNUZ,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F8<const E: usize, const M: usize,
    const N: NanStyle = {NanStyle::IEEE},
    const B: i32 = {(1 << (E - 1)) - 1},
>(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16<const E: usize, const M: usize>(u16);

impl <const E: usize, const M: usize, const N: NanStyle, const B: i32> F8<E, M, N, B> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 8);

    #[must_use]
    pub const fn from_bits(v: u8) -> Self {
        Self(v)
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl <const E: usize, const M: usize> F16<E, M> {
    const _HAS_VALID_STORAGE: () = assert!(E + M < 16);

    #[must_use]
    pub const fn from_bits(v: u16) -> Self {
        Self(v)
    }

    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }
}