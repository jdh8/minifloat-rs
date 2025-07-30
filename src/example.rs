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

crate::minifloat!(pub struct F8E3M4(u8): 3, 4);
crate::minifloat!(pub struct F8E3M4FN(u8): 3, 4, FN);

crate::minifloat!(pub struct F8E4M3(u8): 4, 3);
crate::minifloat!(pub struct F8E4M3FN(u8): 4, 3, FN);
crate::minifloat!(pub struct F8E4M3FNUZ(u8): 4, 3, FNUZ);

crate::minifloat!(pub struct F8E4M3B11(u8): 4, 3, 11);
crate::minifloat!(pub struct F8E4M3B11FN(u8): 4, 3, 11, FN);
crate::minifloat!(pub struct F8E4M3B11FNUZ(u8): 4, 3, 11, FNUZ);

crate::minifloat!(pub struct F8E5M2(u8): 5, 2);
crate::minifloat!(pub struct F8E5M2FNUZ(u8): 5, 2, FNUZ);

crate::minifloat!(pub struct F6E3M2FN(u8): 3, 2, FN);
crate::minifloat!(pub struct F6E2M3FN(u8): 2, 3, FN);
crate::minifloat!(pub struct F4E2M1FN(u8): 2, 1, FN);
