minifloat
=========
[![Build Status](https://github.com/jdh8/minifloat-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/jdh8/minifloat-rs)
[![Crates.io](https://img.shields.io/crates/v/minifloat.svg)](https://crates.io/crates/minifloat)
[![Documentation](https://docs.rs/minifloat/badge.svg)](https://docs.rs/minifloat)

Rust const generic library for minifloats

This crate provides emulation of minifloats up to 16 bits.  This is done with
two generic structs, [`F8`][F8] and [`F16`][F16], which take up to 8 and 16 bits
of storage respectively.  Many parameters are configurable, including

- Exponent width
- Significand (mantissa) precision
- ([`F8`][F8]-only) Exponent bias
- ([`F8`][F8]-only) NaN encodings: IEEE, FN, or FNUZ

Note that there is always a sign bit, so [`F8<4, 3>`][F8] already uses up all 8
bits: 1 sign bit, 4 exponent bits, and 3 significand bits.

[F8]: https://docs.rs/minifloat/latest/minifloat/struct.F8.html
[F16]: https://docs.rs/minifloat/latest/minifloat/struct.F16.html