// This file is part of the minifloat project.
//
// Copyright (C) 2025-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The predicate and comparison layer, timed across a crate boundary
//!
//! Nothing here has an alternative route to be timed against &mdash; a
//! predicate is a handful of bit operations either way.  What this benchmark
//! watches is whether those bit operations reach the caller at all, or whether
//! each one costs a call out of the inlined body around it.  It exists so both
//! sides of an A/B build carry the same bench code.

use criterion::{BenchmarkId, Criterion, Throughput};
use minifloat::{Minifloat, BF16, F16, F8E4M3, F8E5M2FNUZ};

use core::hint::black_box;
use core::time::Duration;

/// Operand pairs one measurement runs over
///
/// Small enough to stay in L1, so the loop times the predicate and not memory.
const PAIRS: usize = 1 << 10;

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
/// have in the format, which is the mix a predicate has to survive.  A unary
/// predicate reads the left operand of each pair.
fn pairs<T: Minifloat>() -> Vec<(T, T)>
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

/// One criterion group per shape, four predicates in each
macro_rules! bench_shapes {
    ($criterion:expr, $($shape:ident)*) => {$({
        let pairs = pairs::<$shape>();
        let mut group = $criterion.benchmark_group(concat!(stringify!($shape), "/predicate"));
        group.throughput(Throughput::Elements(PAIRS as u64));

        group.bench_function(BenchmarkId::from_parameter("is_nan"), |bencher| {
            bencher.iter(|| {
                for &(x, _) in &pairs {
                    black_box(x.is_nan());
                }
            });
        });
        group.bench_function(BenchmarkId::from_parameter("classify"), |bencher| {
            bencher.iter(|| {
                for &(x, _) in &pairs {
                    black_box(x.classify());
                }
            });
        });
        group.bench_function(BenchmarkId::from_parameter("partial_cmp"), |bencher| {
            bencher.iter(|| {
                for &(x, y) in &pairs {
                    black_box(x.partial_cmp(&y));
                }
            });
        });
        group.bench_function(BenchmarkId::from_parameter("total_cmp"), |bencher| {
            bencher.iter(|| {
                for &(x, y) in &pairs {
                    black_box(x.total_cmp(&y));
                }
            });
        });

        group.finish();
    })*};
}

fn bench_predicates(criterion: &mut Criterion) {
    // One shape per thing that can differ: a plain `IEEE` `u8`, the `FNUZ`
    // whose NaN is a bare bit pattern, and both `u16` shapes.
    bench_shapes! { criterion, F8E4M3 F8E5M2FNUZ F16 BF16 }
}

criterion::criterion_group! {
    name = predicates;
    // Each iteration already averages 1024 predicates, so a short window is
    // plenty to settle.  `benches/arith.rs` is configured the same way.
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = bench_predicates
}
criterion::criterion_main!(predicates);
