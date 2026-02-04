import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from dataclasses import dataclass
from collections import defaultdict
import os


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


def list_slower_shapes(results: list[BenchResult]):
    """List all shapes where registered kernel is slower than reference."""
    slower_shapes = []
    
    for r in results:
        if r.registered_time > r.reference_time:
            speedup = r.reference_time / r.registered_time
            slower_shapes.append({
                'm': r.m, 'n': r.n, 'k': r.k, 'l': r.l,
                'reference_time': r.reference_time,
                'registered_time': r.registered_time,
                'speedup': speedup
            })
    
    print("\n" + "=" * 80)
    print("Shapes where Registered kernel is SLOWER than Reference")
    print("=" * 80)
    
    if len(slower_shapes) == 0:
        print("None! All shapes have registered faster than or equal to reference.")
    else:
        print(f"Total: {len(slower_shapes)} / {len(results)} shapes ({100*len(slower_shapes)/len(results):.1f}%)")
        print("-" * 80)
        print(f"{'M':>6} {'N':>6} {'K':>6} {'L':>4} {'Ref (ms)':>10} {'Reg (ms)':>10} {'Speedup':>10}")
        print("-" * 80)
        # Sort by speedup (worst first, i.e., lowest speedup)
        slower_shapes.sort(key=lambda x: x['speedup'])
        for shape in slower_shapes:
            print(f"{shape['m']:>6} {shape['n']:>6} {shape['k']:>6} {shape['l']:>4} "
                  f"{shape['reference_time']:>10.4f} {shape['registered_time']:>10.4f} "
                  f"{shape['speedup']:>9.2f}x")
        print("-" * 80)
    
    return slower_shapes


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


def draw_heatmap(results: list[BenchResult], output_dir: str = "."):
    """Generate heatmap showing speedup for each (m, n) pair with neutral at 1."""
    # Group results by (m, n)
    grouped = defaultdict(list)
    for r in results:
        grouped[(r.m, r.n)].append(r)
    
    # Get unique m and n values
    m_values = sorted(set(r.m for r in results))
    n_values = sorted(set(r.n for r in results))
    
    # Calculate average speedup for each (m, n)
    avg_speedups = np.zeros((len(m_values), len(n_values)))
    
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if (m, n) in grouped:
                ref_times = np.array([r.reference_time for r in grouped[(m, n)]])
                reg_times = np.array([r.registered_time for r in grouped[(m, n)]])
                avg_speedups[i, j] = np.mean(ref_times / reg_times)
            else:
                avg_speedups[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create diverging colormap with neutral at 1.0
    # Red for < 1 (slower), Green for > 1 (faster)
    valid_speedups = avg_speedups[~np.isnan(avg_speedups)]
    if len(valid_speedups) > 0:
        vmin = min(valid_speedups.min(), 0.5)  # Ensure we have room below 1
        vmax = max(valid_speedups.max(), 1.5)  # Ensure we have room above 1
    else:
        vmin, vmax = 0.5, 1.5
    
    # Use TwoSlopeNorm to center the colormap at 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    
    # RdYlGn: Red (low) -> Yellow (mid) -> Green (high)
    im = ax.imshow(avg_speedups, cmap='RdYlGn', aspect='auto', norm=norm)
    
    # Set ticks
    ax.set_xticks(np.arange(len(n_values)))
    ax.set_yticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yticklabels([str(m) for m in m_values])
    
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('M', fontsize=12)
    ax.set_title('Average Speedup (Registered vs Reference)\nAveraged across K values\n(Red < 1x < Green)', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (x)', fontsize=12)
    
    # Annotate cells with values
    for i in range(len(m_values)):
        for j in range(len(n_values)):
            if not np.isnan(avg_speedups[i, j]):
                speedup = avg_speedups[i, j]
                text_color = 'white' if speedup < 0.7 or speedup > 1.5 else 'black'
                text = ax.text(j, i, f'{speedup:.2f}x',
                              ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "summary_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap to: {fig_path}")
    
    return fig_path


def draw_per_mn_plots(results: list[BenchResult], output_dir: str = "."):
    """Generate individual plots for each (m, n) pair."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by (m, n)
    grouped = defaultdict(list)
    for r in results:
        grouped[(r.m, r.n)].append(r)
    
    for (m, n), data in grouped.items():
        if len(data) == 0:
            continue
        
        # Sort by k
        data.sort(key=lambda x: x.k)
        k_values = np.array([r.k for r in data])
        ref_times = np.array([r.reference_time for r in data])
        best_times = np.array([r.best_time for r in data])
        reg_times = np.array([r.registered_time for r in data])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Execution time vs K
        ax1 = axes[0]
        ax1.plot(k_values, ref_times, 'o-', label='Reference', linewidth=2, markersize=6)
        ax1.plot(k_values, best_times, 's-', label='Best (Custom)', linewidth=2, markersize=6)
        ax1.plot(k_values, reg_times, '^-', label='Registered', linewidth=2, markersize=6)
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('K (log2 scale)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title(f'Execution Time vs K\n(M={m}, N={n})', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        ax1.set_xticklabels([str(k) for k in k_values])
        
        # Plot 2: Speedup vs K
        ax2 = axes[1]
        speedup_best = ref_times / best_times
        speedup_reg = ref_times / reg_times
        ax2.plot(k_values, speedup_best, 's-', label='Best vs Reference', linewidth=2, markersize=6, color='green')
        ax2.plot(k_values, speedup_reg, '^-', label='Registered vs Reference', linewidth=2, markersize=6, color='orange')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('K (log2 scale)', fontsize=12)
        ax2.set_ylabel('Speedup (x)', fontsize=12)
        ax2.set_title(f'Speedup vs K\n(M={m}, N={n})', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        ax2.set_xticklabels([str(k) for k in k_values])
        
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f"benchmark_m{m}_n{n}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, help="Input file path")
    parser.add_argument("--output", "-o", type=str, default="bench_results.png", help="Output image path")
    parser.add_argument("--output-dir", "-d", type=str, default="benchmark_figures", help="Output directory for figures")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        text = f.read()

    results = parse_bench_output(text)
    print(f"Parsed {len(results)} benchmark results")
    
    # List all shapes where registered is slower than reference
    slower_shapes = list_slower_shapes(results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Draw heatmap with neutral at 1
    draw_heatmap(results, args.output_dir)
    
    # Draw per (m, n) plots
    draw_per_mn_plots(results, args.output_dir)
    
    # Draw bar chart if few results
    if len(results) <= 20:
        draw_bench_results(results, args.output)
    
    print(f"\nAll figures saved to: {args.output_dir}/")
