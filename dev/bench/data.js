window.BENCHMARK_DATA = {
  "lastUpdate": 1787400531898,
  "repoUrl": "https://github.com/jdh8/minifloat-rs",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "chen.pang.he@jdh8.org",
            "name": "Chen-Pang He",
            "username": "jdh8"
          },
          "committer": {
            "email": "chen.pang.he@jdh8.org",
            "name": "Chen-Pang He",
            "username": "jdh8"
          },
          "distinct": true,
          "id": "72235e3a280642e429148f36bec22bc9c9b33cd5",
          "message": "Say that gh-pages has to exist before the first publish\n\nThe first run failed on `couldn't find remote ref gh-pages` (32475306070):\n`auto-push` fetches the branch and pushes to it, it does not create it.\nThe branch is now an orphan commit over the empty tree, and this push is\nwhat asks the workflow to fill it.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-21T19:09:25+08:00",
          "tree_id": "865fa62bbc7897ee399dddfe90f688a5e4526a58",
          "url": "https://github.com/jdh8/minifloat-rs/commit/72235e3a280642e429148f36bec22bc9c9b33cd5"
        },
        "date": 1787311018167,
        "tool": "cargo",
        "benches": [
          {
            "name": "F4E2M1FN/add/soft",
            "value": 4720,
            "range": "± 117",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/add/f32",
            "value": 5633,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/soft",
            "value": 5494,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/f32",
            "value": 5628,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/soft",
            "value": 3973,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/f32",
            "value": 5248,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/soft",
            "value": 5004,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/f32",
            "value": 5809,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/soft",
            "value": 4662,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/f32",
            "value": 5710,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/soft",
            "value": 5124,
            "range": "± 422",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/f32",
            "value": 5723,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/soft",
            "value": 4134,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/f32",
            "value": 5641,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/soft",
            "value": 5349,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/f32",
            "value": 6013,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/soft",
            "value": 5738,
            "range": "± 283",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/f32",
            "value": 6242,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/soft",
            "value": 5613,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/f32",
            "value": 6183,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/soft",
            "value": 4369,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/f32",
            "value": 6112,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/soft",
            "value": 5740,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/f32",
            "value": 6347,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/soft",
            "value": 4634,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/f32",
            "value": 6069,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/soft",
            "value": 4585,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/f32",
            "value": 5995,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/soft",
            "value": 3996,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/f32",
            "value": 5877,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/soft",
            "value": 5020,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/f32",
            "value": 6305,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/soft",
            "value": 5730,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/f32",
            "value": 6774,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/soft",
            "value": 5499,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/f32",
            "value": 6766,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/soft",
            "value": 4509,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/f32",
            "value": 6425,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/soft",
            "value": 5884,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/f32",
            "value": 6835,
            "range": "± 125",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/soft",
            "value": 5659,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/f32",
            "value": 7453,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/soft",
            "value": 5699,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/f32",
            "value": 7177,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/soft",
            "value": 4674,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/f32",
            "value": 6743,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/soft",
            "value": 6617,
            "range": "± 128",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/f32",
            "value": 7173,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/soft",
            "value": 6495,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/f32",
            "value": 6912,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/soft",
            "value": 6617,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/f32",
            "value": 7269,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/soft",
            "value": 4973,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/f32",
            "value": 6315,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/soft",
            "value": 7938,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/f32",
            "value": 6753,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/soft",
            "value": 6461,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/f32",
            "value": 7090,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/soft",
            "value": 6400,
            "range": "± 359",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/f32",
            "value": 7455,
            "range": "± 82",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/soft",
            "value": 4913,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/f32",
            "value": 6222,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/soft",
            "value": 7333,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/f32",
            "value": 6801,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/soft",
            "value": 7943,
            "range": "± 110",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/f32",
            "value": 8147,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/soft",
            "value": 7638,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/f32",
            "value": 8211,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/soft",
            "value": 4959,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/f32",
            "value": 7493,
            "range": "± 176",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/soft",
            "value": 6954,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/f32",
            "value": 7455,
            "range": "± 760",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/soft",
            "value": 6630,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/f32",
            "value": 7386,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/soft",
            "value": 6181,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/f32",
            "value": 7651,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/soft",
            "value": 5014,
            "range": "± 352",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/f32",
            "value": 7117,
            "range": "± 391",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/soft",
            "value": 7739,
            "range": "± 381",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/f32",
            "value": 6839,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/soft",
            "value": 6162,
            "range": "± 825",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/f32",
            "value": 7244,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/soft",
            "value": 5750,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/f32",
            "value": 7424,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/soft",
            "value": 4960,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/f32",
            "value": 6546,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/soft",
            "value": 6487,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/f32",
            "value": 7064,
            "range": "± 694",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/soft",
            "value": 8403,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/f32",
            "value": 7740,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/soft",
            "value": 8222,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/f32",
            "value": 7642,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/soft",
            "value": 5327,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/f32",
            "value": 7957,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/soft",
            "value": 7615,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/f32",
            "value": 8741,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/soft",
            "value": 7243,
            "range": "± 92",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/f64",
            "value": 6925,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/soft",
            "value": 7446,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/f64",
            "value": 6865,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/soft",
            "value": 5183,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/f64",
            "value": 7399,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/soft",
            "value": 7639,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/f64",
            "value": 8670,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/soft",
            "value": 3384,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/f64",
            "value": 4815,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/soft",
            "value": 3311,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/f64",
            "value": 5053,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/soft",
            "value": 3068,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/f64",
            "value": 5277,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/soft",
            "value": 3698,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/f64",
            "value": 5374,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/is_nan",
            "value": 574,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/classify",
            "value": 1136,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/partial_cmp",
            "value": 1949,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/total_cmp",
            "value": 1218,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/is_nan",
            "value": 642,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/classify",
            "value": 755,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/partial_cmp",
            "value": 1327,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/total_cmp",
            "value": 1216,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/is_nan",
            "value": 643,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/classify",
            "value": 983,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/partial_cmp",
            "value": 1976,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/total_cmp",
            "value": 1369,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/is_nan",
            "value": 574,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/classify",
            "value": 961,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/partial_cmp",
            "value": 2067,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/total_cmp",
            "value": 1368,
            "range": "± 5",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "chen.pang.he@jdh8.org",
            "name": "Chen-Pang He",
            "username": "jdh8"
          },
          "committer": {
            "email": "chen.pang.he@jdh8.org",
            "name": "Chen-Pang He",
            "username": "jdh8"
          },
          "distinct": true,
          "id": "59bd5cc2c594f24c22a00a48a9da8a0995f4e9c5",
          "message": "Release 0.3.0\n\nThree things a publish needs that the round had left undone: the version\nstill said `0.3.0-dev`, the changelog still said `[Unreleased]`, and\nnothing declared a toolchain floor.\n\nThe floor is 1.87.0, which is where `cast_signed` and `cast_unsigned`\nbecame const-stable.  Measured rather than assumed: `cargo +1.85 build\n--release` fails with 220 errors and every one of them is\n`integer_sign_cast`, and `cargo +1.88 build --release` is clean, so no\nother feature in the crate reaches higher.  Without the field a 1.86\ntoolchain reads those 220 errors instead of one sentence about itself.\n\n`exclude = [\"CLAUDE.md\"]` keeps the routing table out of the tarball;\nit addresses whoever is working on the crate, not whoever depends on it.\n`docs/` stays in, since the README links it.\n\nGates before the bump, on a clean tree: `cargo test --release` 32 tests\nand 6 doctests green in 19.8 s, `cargo clippy --all-targets` clean,\n`cargo doc --no-deps` and the `cargo package` verification build clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-08-22T20:01:37+08:00",
          "tree_id": "09f12511851cc1af40b29901677549f6be7f2591",
          "url": "https://github.com/jdh8/minifloat-rs/commit/59bd5cc2c594f24c22a00a48a9da8a0995f4e9c5"
        },
        "date": 1787400530968,
        "tool": "cargo",
        "benches": [
          {
            "name": "F4E2M1FN/add/soft",
            "value": 6672,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/add/f32",
            "value": 6905,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/soft",
            "value": 5735,
            "range": "± 284",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/f32",
            "value": 6893,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/soft",
            "value": 4215,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/f32",
            "value": 6392,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/soft",
            "value": 5942,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/f32",
            "value": 7127,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/soft",
            "value": 6742,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/f32",
            "value": 7010,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/soft",
            "value": 5478,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/f32",
            "value": 7036,
            "range": "± 369",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/soft",
            "value": 4596,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/f32",
            "value": 6850,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/soft",
            "value": 6501,
            "range": "± 51",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/f32",
            "value": 7404,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/soft",
            "value": 7382,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/f32",
            "value": 7117,
            "range": "± 145",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/soft",
            "value": 5836,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/f32",
            "value": 7138,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/soft",
            "value": 4664,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/f32",
            "value": 7025,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/soft",
            "value": 6850,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/f32",
            "value": 7590,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/soft",
            "value": 5146,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/f32",
            "value": 6779,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/soft",
            "value": 5137,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/f32",
            "value": 6792,
            "range": "± 620",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/soft",
            "value": 4356,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/f32",
            "value": 6674,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/soft",
            "value": 5779,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/f32",
            "value": 7100,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/soft",
            "value": 5846,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/f32",
            "value": 7262,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/soft",
            "value": 5810,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/f32",
            "value": 7258,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/soft",
            "value": 4766,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/f32",
            "value": 7237,
            "range": "± 245",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/soft",
            "value": 6483,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/f32",
            "value": 7637,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/soft",
            "value": 6133,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/f32",
            "value": 7427,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/soft",
            "value": 6151,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/f32",
            "value": 7449,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/soft",
            "value": 4886,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/f32",
            "value": 7401,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/soft",
            "value": 6960,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/f32",
            "value": 7902,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/soft",
            "value": 8198,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/f32",
            "value": 7327,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/soft",
            "value": 6603,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/f32",
            "value": 7349,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/soft",
            "value": 5232,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/f32",
            "value": 7196,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/soft",
            "value": 7739,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/f32",
            "value": 7781,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/soft",
            "value": 8158,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/f32",
            "value": 7333,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/soft",
            "value": 6559,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/f32",
            "value": 7508,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/soft",
            "value": 5192,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/f32",
            "value": 7200,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/soft",
            "value": 7941,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/f32",
            "value": 7715,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/soft",
            "value": 6377,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/f32",
            "value": 7600,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/soft",
            "value": 6322,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/f32",
            "value": 7636,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/soft",
            "value": 5188,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/f32",
            "value": 7652,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/soft",
            "value": 6935,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/f32",
            "value": 8018,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/soft",
            "value": 8261,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/f32",
            "value": 7285,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/soft",
            "value": 6619,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/f32",
            "value": 7419,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/soft",
            "value": 5230,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/f32",
            "value": 7229,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/soft",
            "value": 7763,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/f32",
            "value": 7756,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/soft",
            "value": 6308,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/f32",
            "value": 7290,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/soft",
            "value": 6258,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/f32",
            "value": 7413,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/soft",
            "value": 5386,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/f32",
            "value": 7305,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/soft",
            "value": 7198,
            "range": "± 405",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/f32",
            "value": 7726,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/soft",
            "value": 7952,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/f32",
            "value": 8624,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/soft",
            "value": 7854,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/f32",
            "value": 8596,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/soft",
            "value": 5664,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/f32",
            "value": 9844,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/soft",
            "value": 7759,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/f32",
            "value": 11179,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/soft",
            "value": 8123,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/f64",
            "value": 7706,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/soft",
            "value": 7948,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/f64",
            "value": 7513,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/soft",
            "value": 5623,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/f64",
            "value": 7415,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/soft",
            "value": 7640,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/f64",
            "value": 8315,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/soft",
            "value": 3590,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/f64",
            "value": 5071,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/soft",
            "value": 3591,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/f64",
            "value": 5109,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/soft",
            "value": 3314,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/f64",
            "value": 5043,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/soft",
            "value": 4203,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/f64",
            "value": 5420,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/is_nan",
            "value": 556,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/classify",
            "value": 867,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/partial_cmp",
            "value": 1694,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/total_cmp",
            "value": 1003,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/is_nan",
            "value": 597,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/classify",
            "value": 450,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/partial_cmp",
            "value": 1301,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/total_cmp",
            "value": 1003,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/is_nan",
            "value": 595,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/classify",
            "value": 867,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/partial_cmp",
            "value": 1871,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/total_cmp",
            "value": 1005,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/is_nan",
            "value": 595,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/classify",
            "value": 927,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/partial_cmp",
            "value": 1820,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/total_cmp",
            "value": 1005,
            "range": "± 3",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}