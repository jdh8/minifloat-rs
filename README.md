# minifloat-rs
Rust version of [jdh8/minifloat](https://github.com/jdh8/minifloat)

This crate provides emulation of minifloats up to 16 bits.  Many parameters
are configurable, including

- The exponent width
- The mantissa (significand) width
- (F8-only) the bias of the exponent
- (F8-only) NaN encodings: IEEE, FN, or FNUZ