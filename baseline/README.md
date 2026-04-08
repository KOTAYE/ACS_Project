# Performance Baselines

A "Baseline" is a timestamped snapshot of the project's speed (FPS, milliseconds per frame, compression time). We store these snapshots as JSON files so we can objectively compare performance **before** and **after** any code changes.

## Why Baselines Matter

Trusting "it feels faster" is unreliable. Baselines give us hard numbers:
- **Compression FPS** — how many frames the encoder processes per second.
- **Decompression FPS** — how many frames the decoder restores per second.
- **Average ms/frame** — latency for a single frame decode.

Each JSON file also records the **Git revision**, **quality**, **block size**, and **YCbCr mode** so the conditions are always reproducible.

## Directory Structure

```
baseline/
├── README.md          # This file
├── TEMPLATE.json      # Reference schema for a baseline record
└── runs/              # Auto-created by capture_baseline.py
    ├── baseline_20260401T120000Z_abc1234.json
    └── baseline_20260408T183000Z_def5678.json
```

## JSON Schema (TEMPLATE.json)

Every baseline record follows the `flipbook_baseline_v1` schema:

| Field                           | Type   | Description                                    |
|----------------------------------|--------|------------------------------------------------|
| `schema`                        | string | Always `flipbook_baseline_v1`                  |
| `captured_at_utc`               | string | ISO-8601 UTC timestamp of capture              |
| `git_rev_short`                 | string | Short Git hash at the time of capture          |
| `executable`                    | string | Path to the binary used                        |
| `frames_dir`                    | string | Path to the input frames folder                |
| `quality`                       | int    | Quality parameter (1–100)                      |
| `block_size`                    | int    | DCT block size (8, 16, or 32)                  |
| `use_ycbcr`                     | bool   | Whether YCbCr color conversion was enabled     |
| `compress.compress_ms`          | float  | Total compression wall-clock time (ms)         |
| `compress.frames`               | int    | Number of frames compressed                    |
| `compress.compress_fps`         | float  | Throughput during compression (frames/sec)     |
| `decompress.decode_total_ms`    | float  | Total decompression wall-clock time (ms)       |
| `decompress.frames`             | int    | Number of frames decompressed                  |
| `decompress.avg_ms_per_frame`   | float  | Average latency per frame (ms)                 |
| `decompress.decode_fps`         | float  | Throughput during decompression (frames/sec)   |

---

## How to Capture a Baseline

After building in **Release** mode:

```bash
python3 scripts/capture_baseline.py ./Frames -q 50 -b 8
```

**Flags:**
- `--exe <path>` — Path to the executable (default: `build/flipbook_cuda`).
- `-q <1-100>` — Quality level. Default is 50.
- `-b <8|16|32>` — DCT block size. Default is 8.
- `--no-ycbcr` — Skip YCbCr conversion (compress in RGB).
- `--out-dir <path>` — Output directory for JSON (default: `baseline/runs`).

The script runs compress and decompress, parses `[BENCHMARK]` lines from stdout, cleans up temporary files, and saves a JSON record in `baseline/runs/baseline_<UTC>_<git>.json`.

---

## How to Compare Two Baselines

```bash
python3 scripts/compare_baselines.py baseline/runs/old_run.json baseline/runs/new_run.json
```

The script prints **percentage change** for the key metrics:
```
compress_fps: 312.5 → 490.1  +56.83%
compress_ms:  1280.0 → 816.3 -36.23%
decode_fps:   980.2 → 1120.5 +14.31%
avg_ms/frame: 1.02 → 0.89    -12.75%
```

A positive `compress_fps` / `decode_fps` change means the code got faster. A negative `compress_ms` / `avg_ms/frame` change means less time spent — also faster.

---

## Recommended Workflow

1. **Before** changing code — capture a baseline, note the Git revision.
2. **Apply** your optimizations.
3. **After** the changes — capture another baseline.
4. **Compare** the two JSON files with `compare_baselines.py`.
5. **If results are unexpected** — use NVIDIA Nsight tools to investigate (see [PROFILING_AND_BASELINE.md](../docs/PROFILING_AND_BASELINE.md)).

---

## Related Documentation

- **[Profiling & Benchmarking](../docs/PROFILING_AND_BASELINE.md)** — Full guide on Nsight Systems, Nsight Compute, and baseline capture workflow.
- **[Block Size Scaling](../docs/BLOCK_SIZE_SCALING.md)** — How `-b 8|16|32` affects speed and quality.
