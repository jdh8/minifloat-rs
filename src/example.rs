// This file is part of the minifloat project.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example module consisting of generated types
//!
//! This module only serves as an example.  Its content is subject to change.
//! This module is not considered part of the public API.

macro_rules! lossless {
    ($name:ident) => {
        crate::impl_from_minifloat_for_f32!($name);
        crate::impl_from_minifloat_for_f64!($name);
    };
}

crate::minifloat!(pub struct F8E3M4(u8): 3, 4);
lossless!(F8E3M4);
crate::minifloat!(pub struct F8E3M4FN(u8): 3, 4, FN);
lossless!(F8E3M4FN);

crate::minifloat!(pub struct F8E4M3(u8): 4, 3);
lossless!(F8E4M3);
crate::minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
lossless!(F8E4M3FN);
crate::minifloat!(pub struct F8E4M3FNUZ(u8): 4, 3, FNUZ);
lossless!(F8E4M3FNUZ);

crate::minifloat!(pub struct F8E4M3B11(u8): 4, 3, 11);
lossless!(F8E4M3B11);
crate::minifloat!(pub struct F8E4M3B11FN(u8): 4, 3, 11, FN);
lossless!(F8E4M3B11FN);
crate::minifloat!(pub struct F8E4M3B11FNUZ(u8): 4, 3, 11, FNUZ);
lossless!(F8E4M3B11FNUZ);

crate::minifloat!(pub struct F8E5M2(u8): 5, 2);
lossless!(F8E5M2);
crate::minifloat!(pub struct F8E5M2FNUZ(u8): 5, 2, FNUZ);
lossless!(F8E5M2FNUZ);

crate::minifloat!(pub struct F6E3M2FN(u8): 3, 2, FN);
lossless!(F6E3M2FN);
crate::minifloat!(pub struct F6E2M3FN(u8): 2, 3, FN);
lossless!(F6E2M3FN);
crate::minifloat!(pub struct F4E2M1FN(u8): 2, 1, FN);
lossless!(F4E2M1FN);
