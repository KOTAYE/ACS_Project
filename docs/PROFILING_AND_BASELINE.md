# Profiling and Benchmarking Guide

To verify that our optimizations actually work, we use professional profiling tools from NVIDIA and a custom "Baseline" system.

## NVIDIA Nsight Tools

### 1. Nsight Systems (Timeline Analysis)
Use this to see if the GPU is "waiting" for the CPU.

- **Command**:
  ```bash
  bash scripts/profile_nsys.sh
  ```
- **What to look for**: Look for big gaps in the timeline. Ideally, the `cudaMemcpyAsync` (blue) should happen at the same time as `Compute Kernels` (green). This is called "Latency Hiding".

### 2. Nsight Compute (Kernel Analysis)
Use this to see why a specific math operation (like DCT) is slow.

- **Command**:
  ```bash
  bash scripts/profile_ncu.sh
  ```
- **What to look for**: Check **Memory Throughput** and **Compute Throughput**. If memory is 90% and compute is 10%, your kernel is "Memory Bound" (waiting for data). Our DCT kernels are carefully tuned to balance these.

---

## Performance Baselines

A "Baseline" is a snapshot of how slow or fast the project is right now. We save these as JSON files.

### 1. Capture a new baseline
If you make a change and want to see if it helps:
```bash
python3 scripts/capture_baseline.py ./Frames -q 50
```
This saves a file in `baseline/runs/` with details like FPS and memory bandwidth.

### 2. Compare with a previous run
```bash
python3 scripts/compare_baselines.py baseline/runs/old_run.json baseline/runs/new_run.json
```
This script will tell you the percentage of speedup or slowdown.

---

## Troubleshooting Common Issues

- **"NSYS not found"**: Make sure the NVIDIA Nsight Systems directory is in your `PATH`.
- **"Permission Denied"**: You might need to run profiling as `sudo` on Linux to access hardware performance counters.
- **Low Occupancy**: If Nsight Compute says occupancy is low, try changing the block size with `-b 8` or `-b 16` to see how it affects register usage.

## Workflow Summary
- **Step 1**: Run `capture_baseline.py` before you change any code.
- **Step 2**: Apply your optimizations.
- **Step 3**: Run `capture_baseline.py` again.
- **Step 4**: Use `compare_baselines.py` to see the results.
- **Step 5**: If results are weird, use **Nsight** to investigate the hardware behavior.
