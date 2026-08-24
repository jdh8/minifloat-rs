// This file is part of the minifloat project.
//
// Copyright (C) 2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Direct `f32` encoding against the same inputs routed through `f64`

use criterion::{BenchmarkId, Criterion, Throughput};
use minifloat::{BF16, BF20, F16, F8E4M3, F8E5M2FNUZ};

use core::hint::black_box;
use core::time::Duration;

const INPUTS: usize = 1 << 10;

/// Representation-uniform inputs, deterministic and shared by both routes
fn f32s() -> Vec<f32> {
    let mut state = 0x0FED_CBA9u32;
    (0..INPUTS)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits(state)
        })
        .collect()
}

macro_rules! bench_shapes {
    ($criterion:expr, $($shape:ident)*) => {$({
        let inputs = f32s();
        let mut group = $criterion.benchmark_group(concat!(stringify!($shape), "/from_f32"));
        group.throughput(Throughput::Elements(INPUTS as u64));

        group.bench_function(BenchmarkId::from_parameter("direct"), |bencher| {
            bencher.iter(|| {
                for &x in &inputs {
                    black_box($shape::from_f32(x));
                }
            });
        });
        // The historical ID stays stable even when `from_f64` has a fast path.
        group.bench_function(BenchmarkId::from_parameter("generic"), |bencher| {
            bencher.iter(|| {
                for &x in &inputs {
                    black_box($shape::from_f64(f64::from(x)));
                }
            });
        });

        group.finish();
    })*};
}

fn bench_from_f32(criterion: &mut Criterion) {
    bench_shapes! { criterion, BF16 BF20 F16 F8E4M3 F8E5M2FNUZ }
}

criterion::criterion_group! {
    name = from_f32;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = bench_from_f32
}
criterion::criterion_main!(from_f32);
