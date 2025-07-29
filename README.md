# minifloat

[![Build Status](https://github.com/jdh8/minifloat-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/jdh8/minifloat-rs)
[![Crates.io](https://img.shields.io/crates/v/minifloat.svg)](https://crates.io/crates/minifloat)
[![Documentation](https://docs.rs/minifloat/badge.svg)](https://docs.rs/minifloat)

Rust meta-library for minifloats

This crate provides emulation of minifloats up to 16 bits.  Many parameters are
configurable, including

- Exponent width
- Significand (mantissa) precision
- Exponent bias for types up to 8 bits
- NaN encodings for types up to 8 bits: IEEE, FN, or FNUZ
