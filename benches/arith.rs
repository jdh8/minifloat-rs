// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Soft arithmetic against a round trip through a hardware float
//!
//! Each operator is timed twice on the same operands: once as the crate
//! computes it, on integer significands, and once the way a caller would fake
//! it &mdash; widen both operands, let the FPU work, round the result back.
//! [`route`] decides which hardware float a shape is entitled to; a shape no
//! hardware float can round for is skipped rather than compared against a
//! different answer.

use criterion::{BenchmarkId, Criterion, Throughput};
use minifloat::{
    minifloat, Minifloat, BF16, F16, F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3, F8E4M3B11FNUZ,
    F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
};

use core::hint::black_box;
use core::time::Duration;

mod common;
use common::{pairs, PAIRS};

// Shapes reaching past `f32`, the same ones `tests/all/support.rs` sweeps.
// They are what makes [`route`] visible: `E11M4` falls back to `f64`, `E2M13`
// is exact in `f32` yet too wide to round through it, and `E12M3` overruns
// `f64` altogether.
minifloat!(struct E11M4(u16): 11, 4);
minifloat!(struct E2M13(u16): 2, 13);
minifloat!(struct E12M3(u16): 12, 3);

/// The hardware float a shape may be compared against
#[derive(Debug, Clone, Copy)]
enum Route {
    F32,
    F64,
}

/// The narrowest hardware float that rounds every operator like the shape does
///
/// Two things have to hold.  The operands must be exact, or the round trip
/// starts from a different number.  And the intermediate must carry at least
/// 2<var>p</var> + 2 digits, where <var>p</var> is the shape's own precision:
/// below that, rounding to the intermediate and then to the shape can differ
/// from rounding to the shape once (Figueroa 1995).  Exactness alone is not
/// enough &mdash; `E2M13` is exact in `f32`, yet a product of two of its
/// significands is 28 digits wide.
///
/// `None` is a shape no hardware float can referee, which the benchmark skips.
fn route<T: Minifloat>() -> Option<Route> {
    let digits = 2 * T::MANTISSA_DIGITS + 2;

    if T::HAS_EXACT_F32_CONVERSION && digits <= f32::MANTISSA_DIGITS {
        Some(Route::F32)
    } else if T::HAS_EXACT_F64_CONVERSION && digits <= f64::MANTISSA_DIGITS {
        Some(Route::F64)
    } else {
        None
    }
}

/// Time one operator both ways
///
/// The soft route is the operator itself.  The hardware route is whichever
/// widening [`route`] allowed, spelt out per arm so the conversion inlines
/// into the loop instead of going through a function pointer.
macro_rules! bench_op {
    ($group:expr, $shape:ident, $pairs:expr, $route:expr, $name:literal, $op:tt) => {{
        let pairs: &[($shape, $shape)] = $pairs;

        $group.bench_function(BenchmarkId::new($name, "soft"), |bencher| {
            bencher.iter(|| {
                for &(x, y) in pairs {
                    black_box(x $op y);
                }
            });
        });

        match $route {
            Route::F32 => $group.bench_function(BenchmarkId::new($name, "f32"), |bencher| {
                bencher.iter(|| {
                    for &(x, y) in pairs {
                        black_box(<$shape>::from_f32(x.to_f32() $op y.to_f32()));
                    }
                });
            }),
            Route::F64 => $group.bench_function(BenchmarkId::new($name, "f64"), |bencher| {
                bencher.iter(|| {
                    for &(x, y) in pairs {
                        black_box(<$shape>::from_f64(x.to_f64() $op y.to_f64()));
                    }
                });
            }),
        };
    }};
}

/// One criterion group per shape, four operators in each
macro_rules! bench_shapes {
    ($criterion:expr, $($shape:ident)*) => {$(
        if let Some(route) = route::<$shape>() {
            let pairs = pairs::<$shape>();
            let mut group = $criterion.benchmark_group(stringify!($shape));
            group.throughput(Throughput::Elements(PAIRS as u64));

            bench_op!(group, $shape, &pairs, route, "add", +);
            bench_op!(group, $shape, &pairs, route, "sub", -);
            bench_op!(group, $shape, &pairs, route, "mul", *);
            bench_op!(group, $shape, &pairs, route, "div", /);

            group.finish();
        } else {
            println!("{}: skipped, no hardware float rounds like it", stringify!($shape));
        }
    )*};
}

fn bench_arithmetic(criterion: &mut Criterion) {
    bench_shapes! { criterion,
        F4E2M1FN F6E2M3FN F6E3M2FN
        F8E3M4 F8E4M3 F8E4M3FN F8E4M3FNUZ F8E4M3B11FNUZ F8E5M2 F8E5M2FNUZ
        F16 BF16
        E11M4 E2M13 E12M3
    }
}

criterion::criterion_group! {
    name = arithmetic;
    // The default 3 s warm-up plus 5 s measurement over 112 benches is a
    // quarter-hour run.  Each iteration already averages 1024 operations, so
    // a shorter window is plenty to settle.
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = bench_arithmetic
}
criterion::criterion_main!(arithmetic);
