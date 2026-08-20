# The inlining policy

*Every method a dependent can call is concrete, so its body has to be published
on purpose.*

## Every small concrete method carries `#[inline]`

Plain `#[inline]`, never `#[inline(always)]`.  The attribute's job here is to
put the body in the crate metadata where a dependent's compiler can reach it;
whether to actually paste it in is a judgement the inliner makes with
information this crate does not have.  `always` overrules that judgement, and
there is no measurement here that asks for it.

The set, as of e06641b, is everything below — 61 attributes.  Nothing with a
body is left out; the only unmarked items in the crate are the bodiless
declarations in `trait Minifloat` and its two generic default methods.

- the four operators, their four compound forms, and `Neg`;
- the rounding core: `from_parts`, `to_parts`, `add_impl`;
- the raw `from_bits` / `to_bits` pair, and the four conversions `from_f32`,
  `from_f64`, `to_f32`, `to_f64`;
- the two `From` impls that `impl_from_minifloat_for_f32!` and
  `impl_from_minifloat_for_f64!` generate;
- every predicate the `minifloat!` macro generates, plus `classify`, `abs`,
  `integer_decode`, `const_eq`, `const_partial_cmp`, `total_cmp_key`, `huge`;
- the `PartialEq`, `PartialOrd` and `Hash` impls, and every `Minifloat`
  forwarder;
- `Format::has_inf`, `has_nan`, `has_neg_zero`;
- all six kernels in `detail`: `exp2i`, `decompose`, `round_to_scale`, `align`,
  `add_parts`, `div_parts`.

## Generic functions need nothing

A generic function's MIR is in the metadata whether or not anyone asks, because
a dependent has to monomorphize it.  `Minifloat::is_subnormal` and
`Minifloat::is_normal` are trait default methods and are left bare for that
reason — they are the only generic code in the crate, along with the `H` in
`Hash::hash`, whose enclosing impl is marked anyway.  Marking them would be
noise that reads as though it were load-bearing.

## `const fn` publishes MIR, but do not lean on it

A `const fn` also has its MIR in the metadata — const evaluation downstream
needs it — and from `-O2` up, rustc's MIR inliner spends it.  The six kernels in
`detail` are `const fn`, so they were already inlining across a crate boundary
before they were ever marked.  (They are `const fn` for their own reasons; the
metadata is a consequence, not the motive.)

Two reasons the attribute went on anyway.  It is a heuristic, internal to the
compiler and owing nobody a guarantee; and it evaporates below `-O2`, where MIR
inlining is off and `#[inline]` is the only thing that moves a body across a
crate boundary.

## Measured: 2026-08-21

AMD Ryzen 7 8700F, Fedora 44, rustc 1.97.1 (LLVM 22.1.8).  A probe crate with a
path dependency on this one, release profile, no LTO, 34 `extern "C"` entry
points covering every operator, predicate, comparison and conversion.  The
metric is call instructions in the probe's own object file.

| variant | `-O1` | `-O3` |
| --- | --- | --- |
| before this round | 62 | 4 |
| kernels marked only | 46 | 4 |
| predicates marked only | 16 | 0 |
| **both** | **0** | **0** |

Neither half is worth much alone.  That is the same lesson 6520d5a drew from a
stopwatch — `#[inline]` on the public operators alone measured **0.994x**,
fractionally *worse* than leaving it off, because the caller inlines `add` and
the body it inlines still calls out to `from_parts` and `to_parts`, turning one
call into two — except that here it can be read straight off the symbol table
instead of inferred from a ratio.

At `-O3` the whole round comes down to a single method.  Rustc's own small-body
heuristic already carries every other body across and stops at
`PartialOrd::partial_cmp`, whose body is no longer small once
`const_partial_cmp` has been folded into it.  Marking that one method alone
emits code byte-identical to marking all thirty in the predicate row.  The other
twenty-nine are insurance against the heuristic, and rent paid for the `-O1`
column.

## What the timed half was worth

Only the **operator half** of the policy has ever been timed.  6520d5a marked 19
methods — the operators, the compound forms, `Neg`, `from_parts`, `to_parts`,
and the conversions — and measured **1.130x** on the operators and **1.100x**
with the conversions along, min of 9000 rounds across 30 interleaved passes of
alternating builds, against a **0.98x** noise floor.  `benches/arith.rs` agreed
from the other side at 1.126x geomean over its 56 soft benchmarks, every one of
them positive.  The protocol behind those numbers is
[benchmarking.md](benchmarking.md).

The predicate and kernel halves were **never timed** — they were counted.  At
`-O3` there is nothing for a stopwatch to find, which is the whole point of the
table above.

## LTO would have been cheaper, and is not available

`lto = "fat"` measured 1.139x and 1.100x on the same box — the same numbers, for
none of this work.  This crate cannot ask for it.  Cargo reads a `[profile]`
only from the workspace root and ignores one in a dependency's manifest byte for
byte, so a consumer would have to opt in themselves.  Publishing the bodies is
the only lever a library holds.

## What it costs

The operator half cost a probe binary that is mostly this crate **1.8%**
(6520d5a).  The predicate and kernel halves were never sized; every function
they add is a few dozen instructions at most, and the operators share the same
`from_parts` tail, so there is little to duplicate.  If a body ever stops being
small — a table, a loop with a real trip count — it should lose the attribute
rather than keep it out of habit.

## Checking it

There is no probe crate in this repository; it is a scratch crate, rebuilt when
a claim needs checking.  Recreating it takes a minute:

```sh
cargo new --lib probe && cd probe
cargo add minifloat --path /path/to/minifloat-rs
```

Write one `#[no_mangle] pub extern "C" fn` per entry point you care about,
reaching each method by **both** spellings where the crate offers two — the
inherent `x.is_nan()` and the trait `Minifloat::is_nan(x)` resolve to different
functions, and only the second exercises the forwarders.  Then:

```sh
cargo build --release                       # -O3
ar x target/release/libprobe.rlib
objdump -d -C *.o | grep 'call.*minifloat'  # names what did not inline
objdump -d -C *.o | grep -c 'call.*minifloat'

CARGO_PROFILE_RELEASE_OPT_LEVEL=1 cargo build --release --target-dir t1   # -O1
```

Zero is the expected answer at every optimization level that inlines at all —
verified at `-O1` and `-O3`.  At `-O0` LLVM runs no inliner and `#[inline]` is
only a hint, so the count there means nothing.  Use the grep without `-c` to see
which function lost its attribute; with `-c` you get the pass/fail number and no
name.
