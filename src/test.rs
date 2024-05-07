// This file is part of the minifloat project.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#![cfg(test)]

#[test]
fn sanity() {
    let x: f32 = crate::F8::<3, 4>::from_f32(2.0).into();
    assert_eq!(x, 2.0);
}