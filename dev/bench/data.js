window.BENCHMARK_DATA = {
  "lastUpdate": 1787311018655,
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
      }
    ]
  }
}