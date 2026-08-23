// This file is part of the minifloat project.
//
// Copyright (C) 2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Direct `f32` decoding against an exact `f64` round trip

use criterion::{BenchmarkId, Criterion, Throughput};
use minifloat::{BF16, BF20};

use core::hint::black_box;
use core::time::Duration;

mod common;
use common::{pairs, PAIRS};

macro_rules! bench_shapes {
    ($criterion:expr, $($shape:ident)*) => {$({
        let inputs = pairs::<$shape>();
        let mut group = $criterion.benchmark_group(concat!(stringify!($shape), "/to_f32"));
        group.throughput(Throughput::Elements(PAIRS as u64));

        group.bench_function(BenchmarkId::from_parameter("direct"), |bencher| {
            bencher.iter(|| {
                for &(x, _) in &inputs {
                    black_box(x.to_f32());
                }
            });
        });
        group.bench_function(BenchmarkId::from_parameter("generic"), |bencher| {
            bencher.iter(|| {
                for &(x, _) in &inputs {
                    black_box(x.to_f64() as f32);
                }
            });
        });

        group.finish();
    })*};
}

fn bench_to_f32(criterion: &mut Criterion) {
    bench_shapes! { criterion, BF16 BF20 }
}

criterion::criterion_group! {
    name = to_f32;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = bench_to_f32
}
criterion::criterion_main!(to_f32);
