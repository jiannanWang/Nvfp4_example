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
| `eval_do_bench_deepseek.py` | Benchmark script for DeepSeek model shapes |
| `eval_ncu.py` | Script for NCU profiling |
| `plot_bench_results.py` | Parse and visualize benchmark results |
| `utils.py` | Utility functions (seeding, tensor comparison) |
| `task.py` | Type definitions |
| `deepseek_shapes.csv` | Matrix shapes (m, n, k, l) for DeepSeek model benchmarks |

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
1. Runs `eval_do_bench.py` and saves output to `logs/results.txt`
2. Parses results and generates `output.png` with time comparisons

### 3. Run DeepSeek Benchmark

```bash
python eval_do_bench_deepseek.py
```

This script:
1. Reads matrix shapes from `deepseek_shapes.csv`
2. Benchmarks reference, custom, and registered kernels for each shape
3. Checks correctness against the reference implementation
4. Saves results to `benchmark_figures_deepseek/benchmark_results.csv`
5. Prints a summary of successful/failed runs and performance comparisons

## Benchmark Output

The benchmark compares three implementations across different matrix sizes (m, n, k):

- **Reference**: `torch._scaled_mm` baseline
- **Best**: Custom CuTe/CUTLASS kernel (direct call)
- **Registered**: Custom kernel via `torch.library` registration

The plot shows speedup of registered kernel vs reference (e.g., `3.72x`).
