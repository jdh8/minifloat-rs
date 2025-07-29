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

crate::most8!(F8E3M4, 3, 4);
crate::most8!(F8E3M4FN, 3, 4, FN);

crate::most8!(F8E4M3, 4, 3);
crate::most8!(F8E4M3FN, 4, 3, FN);
crate::most8!(F8E4M3FNUZ, 4, 3, FNUZ);

crate::most8!(F8E4M3B11, 4, 3, 11);
crate::most8!(F8E4M3B11FN, 4, 3, 11, FN);
crate::most8!(F8E4M3B11FNUZ, 4, 3, 11, FNUZ);

crate::most8!(F8E5M2, 5, 2);
crate::most8!(F8E5M2FNUZ, 5, 2, FNUZ);

crate::most8!(F6E3M2FN, 3, 2, FN);
crate::most8!(F6E2M3FN, 2, 3, FN);
crate::most8!(F4E2M1FN, 2, 1, FN);
