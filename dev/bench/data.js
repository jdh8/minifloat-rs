window.BENCHMARK_DATA = {
  "lastUpdate": 1787505966183,
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
          "id": "832e903b94860f11c7f47268a09190fef3c8d997",
          "message": "Record the wide-division benchmark",
          "timestamp": "2026-08-23T05:06:04+08:00",
          "tree_id": "3d01806e7f11f802dfec2e09f040af5c60460589",
          "url": "https://github.com/jdh8/minifloat-rs/commit/832e903b94860f11c7f47268a09190fef3c8d997"
        },
        "date": 1787438603822,
        "tool": "cargo",
        "benches": [
          {
            "name": "F4E2M1FN/add/soft",
            "value": 4717,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/add/f32",
            "value": 5712,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/soft",
            "value": 5501,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/f32",
            "value": 5725,
            "range": "± 281",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/soft",
            "value": 3981,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/f32",
            "value": 5284,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/soft",
            "value": 5160,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/f32",
            "value": 5915,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/soft",
            "value": 4658,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/f32",
            "value": 5867,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/soft",
            "value": 5091,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/f32",
            "value": 5834,
            "range": "± 76",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/soft",
            "value": 4143,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/f32",
            "value": 5731,
            "range": "± 135",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/soft",
            "value": 5810,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/f32",
            "value": 6182,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/soft",
            "value": 5735,
            "range": "± 76",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/f32",
            "value": 6197,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/soft",
            "value": 5670,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/f32",
            "value": 6210,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/soft",
            "value": 4368,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/f32",
            "value": 6097,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/soft",
            "value": 6216,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/f32",
            "value": 6501,
            "range": "± 134",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/soft",
            "value": 4626,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/f32",
            "value": 6099,
            "range": "± 131",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/soft",
            "value": 4587,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/f32",
            "value": 6100,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/soft",
            "value": 3991,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/f32",
            "value": 6001,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/soft",
            "value": 5295,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/f32",
            "value": 6406,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/soft",
            "value": 5675,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/f32",
            "value": 6753,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/soft",
            "value": 5700,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/f32",
            "value": 6809,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/soft",
            "value": 4517,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/f32",
            "value": 6449,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/soft",
            "value": 6551,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/f32",
            "value": 6869,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/soft",
            "value": 5568,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/f32",
            "value": 7301,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/soft",
            "value": 5677,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/f32",
            "value": 7158,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/soft",
            "value": 4681,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/f32",
            "value": 6837,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/soft",
            "value": 7508,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/f32",
            "value": 7186,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/soft",
            "value": 6447,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/f32",
            "value": 6724,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/soft",
            "value": 6616,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/f32",
            "value": 6947,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/soft",
            "value": 5002,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/f32",
            "value": 6269,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/soft",
            "value": 8275,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/f32",
            "value": 6760,
            "range": "± 292",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/soft",
            "value": 6460,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/f32",
            "value": 7501,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/soft",
            "value": 6136,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/f32",
            "value": 6996,
            "range": "± 109",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/soft",
            "value": 4921,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/f32",
            "value": 6233,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/soft",
            "value": 8180,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/f32",
            "value": 6812,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/soft",
            "value": 7980,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/f32",
            "value": 8094,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/soft",
            "value": 7480,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/f32",
            "value": 8237,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/soft",
            "value": 4978,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/f32",
            "value": 7562,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/soft",
            "value": 7486,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/f32",
            "value": 7520,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/soft",
            "value": 6627,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/f32",
            "value": 7559,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/soft",
            "value": 6515,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/f32",
            "value": 7808,
            "range": "± 134",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/soft",
            "value": 5099,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/f32",
            "value": 7232,
            "range": "± 114",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/soft",
            "value": 8346,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/f32",
            "value": 6805,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/soft",
            "value": 6111,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/f32",
            "value": 7281,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/soft",
            "value": 5785,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/f32",
            "value": 7311,
            "range": "± 544",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/soft",
            "value": 4943,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/f32",
            "value": 6527,
            "range": "± 183",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/soft",
            "value": 6848,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/f32",
            "value": 7019,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/soft",
            "value": 8315,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/f32",
            "value": 7592,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/soft",
            "value": 8043,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/f32",
            "value": 7558,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/soft",
            "value": 5383,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/f32",
            "value": 8101,
            "range": "± 114",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/soft",
            "value": 7987,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/f32",
            "value": 8989,
            "range": "± 229",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/soft",
            "value": 7963,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/f64",
            "value": 8754,
            "range": "± 100",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/soft",
            "value": 8129,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/f64",
            "value": 8599,
            "range": "± 140",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/soft",
            "value": 5347,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/f64",
            "value": 6828,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/soft",
            "value": 7440,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/f64",
            "value": 7529,
            "range": "± 158",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/soft",
            "value": 7526,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/f64",
            "value": 6848,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/soft",
            "value": 7250,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/f64",
            "value": 6854,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/soft",
            "value": 5161,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/f64",
            "value": 8070,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/soft",
            "value": 8076,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/f64",
            "value": 8505,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/soft",
            "value": 3312,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/f64",
            "value": 4952,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/soft",
            "value": 3329,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/f64",
            "value": 4794,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/soft",
            "value": 3037,
            "range": "± 87",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/f64",
            "value": 5149,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/soft",
            "value": 3999,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/f64",
            "value": 4899,
            "range": "± 31",
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
            "value": 844,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/partial_cmp",
            "value": 1960,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/total_cmp",
            "value": 1218,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/is_nan",
            "value": 344,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/classify",
            "value": 784,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/partial_cmp",
            "value": 1399,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/total_cmp",
            "value": 1217,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/is_nan",
            "value": 574,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/classify",
            "value": 1016,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/partial_cmp",
            "value": 1983,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/total_cmp",
            "value": 1365,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/is_nan",
            "value": 642,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/classify",
            "value": 847,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/partial_cmp",
            "value": 2028,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/total_cmp",
            "value": 1365,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/is_nan",
            "value": 574,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/classify",
            "value": 1026,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/partial_cmp",
            "value": 2053,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/total_cmp",
            "value": 1287,
            "range": "± 7",
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
          "id": "c6dc68dcb830ac895880335ab36e4be0862d0583",
          "message": "Bump to 0.3.1-dev",
          "timestamp": "2026-08-23T18:05:44+08:00",
          "tree_id": "3cc1c8351605af85a2e027a7a065c55461a836f5",
          "url": "https://github.com/jdh8/minifloat-rs/commit/c6dc68dcb830ac895880335ab36e4be0862d0583"
        },
        "date": 1787479977006,
        "tool": "cargo",
        "benches": [
          {
            "name": "F4E2M1FN/add/soft",
            "value": 4908,
            "range": "± 125",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/add/f32",
            "value": 4528,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/soft",
            "value": 3573,
            "range": "± 176",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/f32",
            "value": 4584,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/soft",
            "value": 2442,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/f32",
            "value": 4096,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/soft",
            "value": 3989,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/f32",
            "value": 4619,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/soft",
            "value": 4494,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/f32",
            "value": 4683,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/soft",
            "value": 3907,
            "range": "± 130",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/f32",
            "value": 4620,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/soft",
            "value": 2665,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/f32",
            "value": 4451,
            "range": "± 283",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/soft",
            "value": 5076,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/f32",
            "value": 5066,
            "range": "± 386",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/soft",
            "value": 4984,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/f32",
            "value": 4697,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/soft",
            "value": 3751,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/f32",
            "value": 4804,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/soft",
            "value": 2786,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/f32",
            "value": 4631,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/soft",
            "value": 4806,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/f32",
            "value": 4862,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/soft",
            "value": 3224,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/f32",
            "value": 4605,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/soft",
            "value": 3130,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/f32",
            "value": 5300,
            "range": "± 210",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/soft",
            "value": 2531,
            "range": "± 135",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/f32",
            "value": 4426,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/soft",
            "value": 3809,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/f32",
            "value": 4718,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/soft",
            "value": 3746,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/f32",
            "value": 5102,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/soft",
            "value": 3557,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/f32",
            "value": 4893,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/soft",
            "value": 2881,
            "range": "± 167",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/f32",
            "value": 5462,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/soft",
            "value": 4572,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/f32",
            "value": 4953,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/soft",
            "value": 4022,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/f32",
            "value": 5213,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/soft",
            "value": 4099,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/f32",
            "value": 5154,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/soft",
            "value": 2983,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/f32",
            "value": 4958,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/soft",
            "value": 5087,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/f32",
            "value": 5905,
            "range": "± 216",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/soft",
            "value": 6252,
            "range": "± 175",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/f32",
            "value": 5047,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/soft",
            "value": 4666,
            "range": "± 218",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/f32",
            "value": 5236,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/soft",
            "value": 3168,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/f32",
            "value": 4798,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/soft",
            "value": 6387,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/f32",
            "value": 5167,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/soft",
            "value": 5504,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/f32",
            "value": 4973,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/soft",
            "value": 4354,
            "range": "± 172",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/f32",
            "value": 5048,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/soft",
            "value": 3066,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/f32",
            "value": 4650,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/soft",
            "value": 5568,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/f32",
            "value": 5072,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/soft",
            "value": 4200,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/f32",
            "value": 5971,
            "range": "± 87",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/soft",
            "value": 4431,
            "range": "± 248",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/f32",
            "value": 4954,
            "range": "± 135",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/soft",
            "value": 3088,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/f32",
            "value": 4977,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/soft",
            "value": 5145,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/f32",
            "value": 5440,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/soft",
            "value": 5536,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/f32",
            "value": 5180,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/soft",
            "value": 4376,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/f32",
            "value": 5497,
            "range": "± 265",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/soft",
            "value": 3646,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/f32",
            "value": 5343,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/soft",
            "value": 5421,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/f32",
            "value": 5124,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/soft",
            "value": 4211,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/f32",
            "value": 5175,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/soft",
            "value": 4296,
            "range": "± 287",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/f32",
            "value": 5932,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/soft",
            "value": 3178,
            "range": "± 237",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/f32",
            "value": 4886,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/soft",
            "value": 5208,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/f32",
            "value": 5128,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/soft",
            "value": 5547,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/f32",
            "value": 5398,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/soft",
            "value": 5638,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/f32",
            "value": 5514,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/soft",
            "value": 3358,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/f32",
            "value": 6647,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/soft",
            "value": 6287,
            "range": "± 120",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/f32",
            "value": 8414,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/soft",
            "value": 5657,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/f64",
            "value": 5734,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/soft",
            "value": 5590,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/f64",
            "value": 5870,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/soft",
            "value": 3479,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/f64",
            "value": 5335,
            "range": "± 258",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/soft",
            "value": 5648,
            "range": "± 275",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/f64",
            "value": 5348,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/soft",
            "value": 5337,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/f64",
            "value": 4998,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/soft",
            "value": 5336,
            "range": "± 416",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/f64",
            "value": 4919,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/soft",
            "value": 3397,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/f64",
            "value": 5198,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/soft",
            "value": 5612,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/f64",
            "value": 5754,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/soft",
            "value": 2469,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/f64",
            "value": 3597,
            "range": "± 159",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/soft",
            "value": 2203,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/f64",
            "value": 3487,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/soft",
            "value": 1907,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/f64",
            "value": 3855,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/soft",
            "value": 2837,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/f64",
            "value": 3662,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/add/soft",
            "value": 432,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/add/f64",
            "value": 5646,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/sub/soft",
            "value": 389,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/sub/f64",
            "value": 5614,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/mul/soft",
            "value": 2468,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/mul/f64",
            "value": 5249,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/div/soft",
            "value": 2883,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/div/f64",
            "value": 5234,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/is_nan",
            "value": 296,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/classify",
            "value": 505,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/partial_cmp",
            "value": 1161,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/total_cmp",
            "value": 750,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/is_nan",
            "value": 255,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/classify",
            "value": 526,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/partial_cmp",
            "value": 1003,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/total_cmp",
            "value": 829,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/is_nan",
            "value": 296,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/classify",
            "value": 628,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/partial_cmp",
            "value": 1089,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/total_cmp",
            "value": 744,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/is_nan",
            "value": 318,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/classify",
            "value": 609,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/partial_cmp",
            "value": 1182,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/total_cmp",
            "value": 744,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/is_nan",
            "value": 296,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/classify",
            "value": 574,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/partial_cmp",
            "value": 1210,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/total_cmp",
            "value": 752,
            "range": "± 2",
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
          "id": "102d57aec8bacbd6df1f3e778c53a11986067ca0",
          "message": "Clean up test and benchmark lints",
          "timestamp": "2026-08-23T19:14:27+08:00",
          "tree_id": "c0728a07255ac5031f4c671c33edfb4cd0bb57f8",
          "url": "https://github.com/jdh8/minifloat-rs/commit/102d57aec8bacbd6df1f3e778c53a11986067ca0"
        },
        "date": 1787485298060,
        "tool": "cargo",
        "benches": [
          {
            "name": "F4E2M1FN/add/soft",
            "value": 4706,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/add/f32",
            "value": 5720,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/soft",
            "value": 5488,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/f32",
            "value": 5700,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/soft",
            "value": 3990,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/f32",
            "value": 5313,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/soft",
            "value": 5151,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/f32",
            "value": 5960,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/soft",
            "value": 4695,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/f32",
            "value": 5816,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/soft",
            "value": 5134,
            "range": "± 564",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/f32",
            "value": 5820,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/soft",
            "value": 4144,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/f32",
            "value": 5729,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/soft",
            "value": 5795,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/f32",
            "value": 6135,
            "range": "± 150",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/soft",
            "value": 5728,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/f32",
            "value": 6250,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/soft",
            "value": 5614,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/f32",
            "value": 6222,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/soft",
            "value": 4332,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/f32",
            "value": 6069,
            "range": "± 82",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/soft",
            "value": 6189,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/f32",
            "value": 6509,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/soft",
            "value": 4619,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/f32",
            "value": 6067,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/soft",
            "value": 4497,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/f32",
            "value": 6063,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/soft",
            "value": 3981,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/f32",
            "value": 6177,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/soft",
            "value": 5350,
            "range": "± 562",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/f32",
            "value": 6376,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/soft",
            "value": 5709,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/f32",
            "value": 6833,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/soft",
            "value": 5502,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/f32",
            "value": 6811,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/soft",
            "value": 4534,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/f32",
            "value": 6438,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/soft",
            "value": 6263,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/f32",
            "value": 6896,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/soft",
            "value": 5723,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/f32",
            "value": 7223,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/soft",
            "value": 5694,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/f32",
            "value": 7291,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/soft",
            "value": 4715,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/f32",
            "value": 6670,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/soft",
            "value": 7291,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/f32",
            "value": 7421,
            "range": "± 230",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/soft",
            "value": 6482,
            "range": "± 87",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/f32",
            "value": 7444,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/soft",
            "value": 6137,
            "range": "± 582",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/f32",
            "value": 7265,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/soft",
            "value": 4967,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/f32",
            "value": 6329,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/soft",
            "value": 8418,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/f32",
            "value": 6782,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/soft",
            "value": 6491,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/f32",
            "value": 7405,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/soft",
            "value": 6414,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/f32",
            "value": 7307,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/soft",
            "value": 4950,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/f32",
            "value": 6235,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/soft",
            "value": 8474,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/f32",
            "value": 6790,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/soft",
            "value": 7988,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/f32",
            "value": 8181,
            "range": "± 321",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/soft",
            "value": 7687,
            "range": "± 100",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/f32",
            "value": 8116,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/soft",
            "value": 5014,
            "range": "± 147",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/f32",
            "value": 7572,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/soft",
            "value": 7470,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/f32",
            "value": 7496,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/soft",
            "value": 6604,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/f32",
            "value": 7380,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/soft",
            "value": 6319,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/f32",
            "value": 7906,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/soft",
            "value": 5045,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/f32",
            "value": 7190,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/soft",
            "value": 7948,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/f32",
            "value": 6812,
            "range": "± 98",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/soft",
            "value": 6122,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/f32",
            "value": 7399,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/soft",
            "value": 5870,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/f32",
            "value": 7378,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/soft",
            "value": 4941,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/f32",
            "value": 6524,
            "range": "± 201",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/soft",
            "value": 6880,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/f32",
            "value": 7093,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/soft",
            "value": 8230,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/f32",
            "value": 7713,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/soft",
            "value": 8096,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/f32",
            "value": 7678,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/soft",
            "value": 5449,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/f32",
            "value": 7926,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/soft",
            "value": 7896,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/f32",
            "value": 8593,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/soft",
            "value": 8108,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/f64",
            "value": 8783,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/soft",
            "value": 7959,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/f64",
            "value": 8634,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/soft",
            "value": 5236,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/f64",
            "value": 6816,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/soft",
            "value": 7100,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/f64",
            "value": 7406,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/soft",
            "value": 7492,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/f64",
            "value": 6904,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/soft",
            "value": 7246,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/f64",
            "value": 6780,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/soft",
            "value": 5155,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/f64",
            "value": 8174,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/soft",
            "value": 8173,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/f64",
            "value": 8981,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/soft",
            "value": 3388,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/f64",
            "value": 4860,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/soft",
            "value": 3358,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/f64",
            "value": 4606,
            "range": "± 190",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/soft",
            "value": 3052,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/f64",
            "value": 5120,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/soft",
            "value": 3961,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/f64",
            "value": 5249,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/add/soft",
            "value": 716,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/add/f64",
            "value": 8325,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/sub/soft",
            "value": 644,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/sub/f64",
            "value": 8399,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/mul/soft",
            "value": 1274,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/mul/f64",
            "value": 6461,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/div/soft",
            "value": 2224,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/div/f64",
            "value": 7095,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/is_nan",
            "value": 574,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/classify",
            "value": 1121,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/partial_cmp",
            "value": 1927,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/total_cmp",
            "value": 1220,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/is_nan",
            "value": 642,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/classify",
            "value": 756,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/partial_cmp",
            "value": 1327,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/total_cmp",
            "value": 1219,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/is_nan",
            "value": 642,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/classify",
            "value": 983,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/partial_cmp",
            "value": 1974,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/total_cmp",
            "value": 1371,
            "range": "± 8",
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
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/partial_cmp",
            "value": 2071,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/total_cmp",
            "value": 1370,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/is_nan",
            "value": 642,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/classify",
            "value": 802,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/partial_cmp",
            "value": 2033,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/total_cmp",
            "value": 1265,
            "range": "± 16",
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
          "id": "595048c056173acca7681da37bf4dacd8ce199e4",
          "message": "Document matching-exponent conversion speedups",
          "timestamp": "2026-08-24T01:12:04+08:00",
          "tree_id": "ea56f0a6c1d024e5c989dfb9738e9d6a5e57cbb1",
          "url": "https://github.com/jdh8/minifloat-rs/commit/595048c056173acca7681da37bf4dacd8ce199e4"
        },
        "date": 1787505965244,
        "tool": "cargo",
        "benches": [
          {
            "name": "F4E2M1FN/add/soft",
            "value": 5345,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/add/f32",
            "value": 6391,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/soft",
            "value": 6057,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/sub/f32",
            "value": 6368,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/soft",
            "value": 4415,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/mul/f32",
            "value": 5901,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/soft",
            "value": 5776,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F4E2M1FN/div/f32",
            "value": 6596,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/soft",
            "value": 5344,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/add/f32",
            "value": 6481,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/soft",
            "value": 5728,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/sub/f32",
            "value": 6520,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/soft",
            "value": 4584,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/mul/f32",
            "value": 6376,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/soft",
            "value": 6555,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F6E2M3FN/div/f32",
            "value": 6816,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/soft",
            "value": 6375,
            "range": "± 309",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/add/f32",
            "value": 6774,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/soft",
            "value": 6252,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/sub/f32",
            "value": 6796,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/soft",
            "value": 4891,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/mul/f32",
            "value": 6672,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/soft",
            "value": 6905,
            "range": "± 457",
            "unit": "ns/iter"
          },
          {
            "name": "F6E3M2FN/div/f32",
            "value": 7134,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/soft",
            "value": 5282,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/add/f32",
            "value": 6593,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/soft",
            "value": 5134,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/sub/f32",
            "value": 6590,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/soft",
            "value": 4477,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/mul/f32",
            "value": 6465,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/soft",
            "value": 5940,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E3M4/div/f32",
            "value": 6923,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/soft",
            "value": 5964,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/add/f32",
            "value": 7018,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/soft",
            "value": 5812,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/sub/f32",
            "value": 6890,
            "range": "± 146",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/soft",
            "value": 5019,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/mul/f32",
            "value": 6873,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/soft",
            "value": 6679,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/div/f32",
            "value": 7398,
            "range": "± 377",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/soft",
            "value": 6327,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/add/f32",
            "value": 7441,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/soft",
            "value": 6353,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/sub/f32",
            "value": 7436,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/soft",
            "value": 5193,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/mul/f32",
            "value": 7386,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/soft",
            "value": 7104,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FN/div/f32",
            "value": 7857,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/soft",
            "value": 7227,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/add/f32",
            "value": 6756,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/soft",
            "value": 6874,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/sub/f32",
            "value": 6765,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/soft",
            "value": 5397,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/mul/f32",
            "value": 6702,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/soft",
            "value": 8185,
            "range": "± 59",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3FNUZ/div/f32",
            "value": 7256,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/soft",
            "value": 7232,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/add/f32",
            "value": 6759,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/soft",
            "value": 6858,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/sub/f32",
            "value": 6743,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/soft",
            "value": 5375,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/mul/f32",
            "value": 6636,
            "range": "± 265",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/soft",
            "value": 8305,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3B11FNUZ/div/f32",
            "value": 7252,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/soft",
            "value": 6450,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/add/f32",
            "value": 7182,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/soft",
            "value": 6302,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/sub/f32",
            "value": 7358,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/soft",
            "value": 5325,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/mul/f32",
            "value": 7347,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/soft",
            "value": 7106,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2/div/f32",
            "value": 7755,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/soft",
            "value": 7199,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/add/f32",
            "value": 6737,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/soft",
            "value": 6930,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/sub/f32",
            "value": 6735,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/soft",
            "value": 5408,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/mul/f32",
            "value": 6679,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/soft",
            "value": 8104,
            "range": "± 408",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/div/f32",
            "value": 7188,
            "range": "± 496",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/soft",
            "value": 6638,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "F16/add/f32",
            "value": 7143,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/soft",
            "value": 6533,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "F16/sub/f32",
            "value": 7147,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/soft",
            "value": 5567,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F16/mul/f32",
            "value": 7060,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/soft",
            "value": 7449,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "F16/div/f32",
            "value": 7618,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/soft",
            "value": 8258,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/add/f32",
            "value": 1401,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/soft",
            "value": 8061,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/sub/f32",
            "value": 1407,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/soft",
            "value": 5788,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/mul/f32",
            "value": 2186,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/soft",
            "value": 7819,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/div/f32",
            "value": 2836,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/soft",
            "value": 7859,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/add/f64",
            "value": 7518,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/soft",
            "value": 7867,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/sub/f64",
            "value": 7527,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/soft",
            "value": 5548,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/mul/f64",
            "value": 7214,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/soft",
            "value": 7348,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/div/f64",
            "value": 7911,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/soft",
            "value": 8622,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/add/f64",
            "value": 1344,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/soft",
            "value": 8252,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/sub/f64",
            "value": 1273,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/soft",
            "value": 5793,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/mul/f64",
            "value": 2158,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/soft",
            "value": 7836,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "E11M4/div/f64",
            "value": 3592,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/soft",
            "value": 3747,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/add/f64",
            "value": 4869,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/soft",
            "value": 3702,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/sub/f64",
            "value": 4854,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/soft",
            "value": 3375,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/mul/f64",
            "value": 4927,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/soft",
            "value": 4460,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "E2M13/div/f64",
            "value": 5197,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/add/soft",
            "value": 809,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/add/f64",
            "value": 7225,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/sub/soft",
            "value": 812,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/sub/f64",
            "value": 7241,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/mul/soft",
            "value": 1409,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/mul/f64",
            "value": 7038,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/div/soft",
            "value": 2498,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "E8M23/div/f64",
            "value": 7566,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/from_f32/direct",
            "value": 954,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/from_f32/generic",
            "value": 4718,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/from_f32/direct",
            "value": 954,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/from_f32/generic",
            "value": 4725,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/is_nan",
            "value": 611,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/classify",
            "value": 1081,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/partial_cmp",
            "value": 1980,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "F8E4M3/predicate/total_cmp",
            "value": 1181,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/is_nan",
            "value": 725,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/classify",
            "value": 545,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/partial_cmp",
            "value": 1369,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "F8E5M2FNUZ/predicate/total_cmp",
            "value": 1181,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/is_nan",
            "value": 726,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/classify",
            "value": 908,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/partial_cmp",
            "value": 2027,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "F16/predicate/total_cmp",
            "value": 1361,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/is_nan",
            "value": 610,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/classify",
            "value": 917,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/partial_cmp",
            "value": 2086,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/predicate/total_cmp",
            "value": 1361,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/is_nan",
            "value": 725,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/classify",
            "value": 772,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/partial_cmp",
            "value": 2108,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/predicate/total_cmp",
            "value": 1280,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/to_f32/direct",
            "value": 443,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "BF16/to_f32/generic",
            "value": 1849,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/to_f32/direct",
            "value": 438,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "BF20/to_f32/generic",
            "value": 1561,
            "range": "± 127",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}