import re
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class BenchResult:
    m: int
    n: int
    k: int
    l: int
    reference_time: float
    best_time: float
    registered_time: float
    registration_overhead: float


def parse_bench_output(text: str) -> list[BenchResult]:
    """Parse benchmark output text into structured results."""
    results = []
    lines = text.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(r'm=(\d+),\s*n=(\d+),\s*k=(\d+),\s*l=(\d+)', line)
        if match:
            m, n, k, l = map(int, match.groups())
            ref_time = float(re.search(r'Reference time:\s*([\d.]+)', lines[i+1]).group(1))
            best_time = float(re.search(r'Best time:\s*([\d.]+)', lines[i+2]).group(1))
            reg_time = float(re.search(r'Registered time:\s*([\d.]+)', lines[i+3]).group(1))
            overhead = float(re.search(r'Registration overhead:\s*([\d.-]+)', lines[i+4]).group(1))
            results.append(BenchResult(m, n, k, l, ref_time, best_time, reg_time, overhead))
            i += 5
        else:
            i += 1
    return results


def draw_bench_results(results: list[BenchResult], output_path: str = "bench_results.png"):
    """Draw bar chart comparing times across different matrix sizes."""
    labels = [f"m={r.m}\nn={r.n}\nk={r.k}" for r in results]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ref_times = [r.reference_time for r in results]
    best_times = [r.best_time for r in results]
    reg_times = [r.registered_time for r in results]

    speedups = [r.reference_time / r.registered_time for r in results]

    bars1 = ax.bar(x - width, ref_times, width, label='Reference', color='#2196F3')
    bars2 = ax.bar(x, best_times, width, label='Best', color='#4CAF50')
    bars3 = ax.bar(x + width, reg_times, width, label='Registered', color='#FF9800')

    ax.set_ylabel('Time (ms)')
    ax.set_xlabel('Matrix Size')
    ax.set_title('NVFP4 Benchmark: Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    def add_labels(bars, speedup_list=None):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label = f'{height:.3f}'
            if speedup_list:
                label += f'\n({speedup_list[i]:.2f}x)'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3, speedups)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="Input file path")
    parser.add_argument("--output", "-o", type=str, default="bench_results.png", help="Output image path")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        text = f.read()

    results = parse_bench_output(text)
    draw_bench_results(results, args.output)
