minifloat
=========
[![Build Status](https://github.com/jdh8/minifloat-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/jdh8/minifloat-rs)
[![Crates.io](https://img.shields.io/crates/v/minifloat.svg)](https://crates.io/crates/minifloat)
[![Documentation](https://docs.rs/minifloat/badge.svg)](https://docs.rs/minifloat)

Rust const generic library for minifloats

This crate provides emulation of minifloats up to 16 bits.  This is done with
two generic structs, [`Most8`][Most8] and [`Most16`][Most16], which take up to 8 and 16 bits
of storage respectively.  Many parameters are configurable, including

- Exponent width
- Significand (mantissa) precision
- ([`Most8`][Most8]-only) Exponent bias
- ([`Most8`][Most8]-only) NaN encodings: IEEE, FN, or FNUZ

Note that there is always a sign bit, so [`Most8<4, 3>`][Most8] already uses up all 8
bits: 1 sign bit, 4 exponent bits, and 3 significand bits.

[Most8]: https://docs.rs/minifloat/latest/minifloat/struct.Most8.html
[Most16]: https://docs.rs/minifloat/latest/minifloat/struct.Most16.html