# NVFP4 GEMM Benchmark Example

Benchmark suite for evaluating NVFP4 (4-bit floating point) block-scaled GEMM kernels using CuTe/CUTLASS.

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

| File | Description |
|------|-------------|
| `best_submission.py` | Custom NVFP4 GEMM kernel using CuTe/CUTLASS |
| `reference.py` | Reference implementation using `torch._scaled_mm` |
| `eval_do_bench.py` | Benchmark script using Triton's `do_bench` |
| `eval_ncu.py` | Script for NCU profiling |
| `plot_bench_results.py` | Parse and visualize benchmark results |
| `utils.py` | Utility functions (seeding, tensor comparison) |
| `task.py` | Type definitions |

## Usage

### 1. NCU Profiling

```bash
bash run_ncu.sh
```

Open the generated `full_profile.ncu-rep` in NVIDIA Nsight Compute GUI for analysis.

### 2. Run Benchmark & Plot Results

```bash
bash run_overhead.sh
```

This script:
1. Runs `eval_do_bench.py` and saves output to `results.txt`
2. Parses results and generates `output.png` with time comparisons


## Benchmark Output

The benchmark compares three implementations across different matrix sizes (m, n, k):

- **Reference**: `torch._scaled_mm` baseline
- **Best**: Custom CuTe/CUTLASS kernel (direct call)
- **Registered**: Custom kernel via `torch.library` registration

The plot shows speedup of registered kernel vs reference (e.g., `3.72x`).
