# Working on minifloat

*The standing decisions live in `docs/`; this file is the routing table to them
and the short list of things that will bite you.*

## Where to read before you touch something

| Touching | Read first |
| --- | --- |
| `src/detail.rs`, any operator, rounding, `from_parts` / `to_parts` | [docs/arithmetic.md](docs/arithmetic.md) |
| `tests/all/arith.rs`, the oracle, the reference encoder | [docs/arithmetic.md](docs/arithmetic.md) |
| an `#[inline]`, anything about what a dependent crate sees | [docs/inlining.md](docs/inlining.md) |
| `benches/`, or any claim with a number in it | [docs/benchmarking.md](docs/benchmarking.md) |
| the `minifloat!` macro's surface, a new format, a new type | [README.md](README.md) |
| finishing a round | [CHANGELOG.md](CHANGELOG.md), its own final commit |

`README.md` is also the crate's rustdoc landing page — `src/lib.rs` pulls it in
with `#![doc = include_str!("../README.md")]` — so an edit there ships to
docs.rs.  The files under `docs/` do not; they are for whoever is working on the
crate.

Those documents record decisions that have already been made and paid for.
Reversing one is fine; reversing one without reading why it was made is how the
same question gets asked a fourth time.

## Standing rules

**Develop on `main`.**  No feature branches in this repository.

**Never reformat the repository.**  `HEAD` is not rustfmt-clean — eight of its
ten source files would change, and `tests/all/arith.rs:177` is 113 columns
against a default `max_width` of 100 — so a repo-wide `cargo fmt` would bury
every real change in noise.  `cargo fmt` also has no way to narrow to one file:
a bare path is rejected, and `cargo fmt --check -- path.rs` checks that path
*plus* every file in the crate.  Call rustfmt directly on a file you just wrote:

```sh
rustfmt --check --edition 2021 path/to/new_file.rs
```

**`cargo clippy --all-targets` stays clean.**  It is clean now; keep it that way
rather than adding an `#[allow]`.  CI does not run it — CI is `cargo build
--release` and `cargo test --release` and nothing else — so it is on you.

**Benchmark only on an idle box.**  The box runs a poker solver that will take
every core.  `pgrep -af poker` first, and if it is running, ask — never kill it
unasked.  The protocol is [docs/benchmarking.md](docs/benchmarking.md), and it
is not optional: interleaved builds, min-of-N across at least 15 alternating
passes each, and a control route.  The control's own spread *is* the noise
floor for that build pair — calibrate it every round rather than assuming
0.98x, and check in the symbol table that the control's code is actually
unchanged.  Relocation alone moved byte-identical rows 9% on 2026-08-21.

## The correctness gate

`cargo test --release` before anything else, and before any benchmarking.  It
takes about 40 seconds and it has caught every arithmetic mistake so far.  The
debug build is slow enough to be worth avoiding for these.

Two distinct gates live in there, and it is worth not conflating them:

- **the exact integer oracle** (`test_arithmetic_correctly_rounded`) — every
  ordered pair of *finite* operands of all 34 shapes in the test roster, refereed
  in `i128` with no float involved;
- **the `f32` round-trip sweep** (`test_arithmetic_matches_f32_16bit`) — all
  2<sup>32</sup> ordered pairs of `F16` and `BF16`, which is what licenses
  `benches/arith.rs` to time the two routes against each other.

The non-finite pairs the oracle skips are pinned by
`test_arithmetic_matches_f64`.  [docs/arithmetic.md](docs/arithmetic.md) has the
full picture.

## Commit conventions

Subjects are imperative and sentence-shaped — *Inline the kernels the operators
are built from*, not *feat: add inline attrs*.  Every measured claim goes in the
commit body with the binary or benchmark that produced it, and a null result is
recorded as plainly as a win.  The changelog gets its own final commit per
round, in Keep a Changelog form.

## Voice

Documentation and comments explain the decision, not the syntax.  Prefer symbol
names over line numbers, state the thesis before the reasoning, and let a
measurement carry its own date and provenance.  A comment that says what the
next line does is worth deleting; one that says why the obvious thing was not
done is worth keeping.
