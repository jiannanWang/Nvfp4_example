import argparse
import csv
import os
from collections import defaultdict
import torch
import torch.cuda
import torch.cuda.nvtx
import triton
import matplotlib.pyplot as plt
import numpy as np

from utils import set_seed
from reference import generate_input, to_blocked, ref_kernel
from best_submission import custom_kernel, compile_kernel
from best_submission_fix import custom_kernel, compile_kernel

LIB_DEF = torch.library.Library("nvfp4_bench", "DEF")
LIB_IMPL = torch.library.Library("nvfp4_bench", "IMPL")

def _setup_registered_kernel(custom_kernel, compile_kernel):
    """Register custom kernel using torch.library."""
    global LIB_DEF, LIB_IMPL

    LIB_DEF.define(
        "_scaled_mm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "Tensor sfa_perm, Tensor sfb_perm, Tensor c) -> Tensor"
    )

    @torch.library.impl(LIB_IMPL, "_scaled_mm", "CUDA")
    def impl(a, b, sfa, sfb, sfa_perm, sfb_perm, c):
        data = (a, b, sfa, sfb, sfa_perm, sfb_perm, c)
        return custom_kernel(data)

    print("  Registered nvfp4_bench::_scaled_mm")


def read_shapes_from_csv(csv_path):
    """Read m, n, k shapes from CSV file."""
    shapes = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row['m'])
            n = int(row['n'])
            k = int(row['k'])
            shapes.append((m, n, k, 1))  # l=1 as default
    return shapes


def load_results_from_csv(csv_path):
    """Load benchmark results from a CSV file."""
    results = defaultdict(lambda: {'k': [], 'reference': [], 'best': [], 'registered': [], 'correct_best': [], 'correct_reg': []})
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m, n, k = int(row['m']), int(row['n']), int(row['k'])
            results[(m, n)]['k'].append(k)
            results[(m, n)]['reference'].append(float(row['reference_time_ms']))
            results[(m, n)]['best'].append(float(row['best_time_ms']))
            results[(m, n)]['registered'].append(float(row['registered_time_ms']))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_results', type=str, default=None,
                        help='Path to benchmark_results.csv to load instead of running benchmarks')
    args = parser.parse_args()

    seed = 1111
    output_dir = "benchmark_figures_deepseek"
    shapes_file = "deepseek_shapes.csv"
    os.makedirs(output_dir, exist_ok=True)

    if args.benchmark_results:
        # Load results from CSV
        print(f"Loading results from: {args.benchmark_results}")
        results = load_results_from_csv(args.benchmark_results)
        m_values = sorted(set(m for m, n in results.keys()))
        n_values = sorted(set(n for m, n in results.keys()))
    else:
        # Run benchmarks
        set_seed(seed)
        _setup_registered_kernel(custom_kernel, compile_kernel)

        # Read shapes from CSV file
        args_dict = read_shapes_from_csv(shapes_file)
        
        # Get unique m and n values for heatmap
        m_values = sorted(set(m for m, n, k, l in args_dict))
        n_values = sorted(set(n for m, n, k, l in args_dict))

        print(f"Total configurations to benchmark: {len(args_dict)}")
        print(f"Unique M values: {len(m_values)}")
        print(f"Unique N values: {len(n_values)}")
        
        # Store results: {(m, n): {'k': [], 'ref': [], 'best': [], 'registered': []}}
        results = defaultdict(lambda: {'k': [], 'reference': [], 'best': [], 'registered': [], 'correct_best': [], 'correct_reg': []})
        
        successful = 0
        failed = 0
        
        for m, n, k, l in args_dict: 
            try:
                data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
                torch.cuda.synchronize()

                reference_time = triton.testing.do_bench(lambda: ref_kernel(data))
                result_ref = ref_kernel(data)
                torch.cuda.synchronize()

                best_time = triton.testing.do_bench(lambda: custom_kernel(data))
                result_best = custom_kernel(data)
                torch.cuda.synchronize()

                registered_time = triton.testing.do_bench(lambda: torch.ops.nvfp4_bench._scaled_mm(*data))
                result_reg = torch.ops.nvfp4_bench._scaled_mm(*data)
                torch.cuda.synchronize()

                # Check correctness
                if not torch.allclose(result_ref, result_best):
                    print(f"m={m}, n={n}, k={k}, l={l} - FAILED: Best result does not match reference!")
                    # print L1 distance
                    print("Diff: ", torch.norm(result_ref - result_best))
                    correct_best = False
                else:
                    correct_best = True
                if not torch.allclose(result_ref, result_reg):
                    print(f"m={m}, n={n}, k={k}, l={l} - FAILED: Registered result does not match reference!")
                    print("Diff: ", torch.norm(result_ref - result_reg))
                    correct_reg = False
                else:
                    correct_reg = True

                print(f"m={m}, n={n}, k={k}, l={l}")
                print(f"  Reference time: {reference_time:.4f} ms")
                print(f"  Best time: {best_time:.4f} ms")
                print(f"  Registered time: {registered_time:.4f} ms")
                print(f"  Registration overhead: {registered_time - best_time:.4f} ms")
                
                # Store results
                results[(m, n)]['correct_best'].append(correct_best)
                results[(m, n)]['correct_reg'].append(correct_reg)
                results[(m, n)]['k'].append(k)
                results[(m, n)]['reference'].append(reference_time)
                results[(m, n)]['best'].append(best_time)
                results[(m, n)]['registered'].append(registered_time)
                
                successful += 1
            except Exception as e:
                print(f"m={m}, n={n}, k={k}, l={l} - FAILED: {type(e).__name__}: {e}")
                failed += 1
                continue
        
        # save results to CSV
        save_results_to_csv(results, output_dir)
        
        print(f"\n=== Summary ===")
        print(f"Successful: {successful}/{len(args_dict)}")
        print(f"Failed: {failed}/{len(args_dict)}")
    
    # Reorganize results by (n, k) for figures with m on x-axis
    results_by_nk = defaultdict(lambda: {'m': [], 'reference': [], 'best': [], 'registered': []})
    for (m, n), data in results.items():
        for i, k in enumerate(data['k']):
            results_by_nk[(n, k)]['m'].append(m)
            results_by_nk[(n, k)]['reference'].append(data['reference'][i])
            results_by_nk[(n, k)]['best'].append(data['best'][i])
            results_by_nk[(n, k)]['registered'].append(data['registered'][i])
    
    # Generate figures for each (n, k) pair with m on x-axis
    print(f"\n=== Generating Figures (fixed N, K; M on x-axis) ===")
    for (n, k), data in results_by_nk.items():
        if len(data['m']) == 0:
            continue
            
        # Sort by m for proper line plots
        sorted_indices = np.argsort(data['m'])
        m_vals = np.array(data['m'])[sorted_indices]
        ref_times = np.array(data['reference'])[sorted_indices]
        best_times = np.array(data['best'])[sorted_indices]
        reg_times = np.array(data['registered'])[sorted_indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Execution time vs M
        ax1 = axes[0]
        ax1.plot(m_vals, ref_times, 'o-', label='Reference', linewidth=2, markersize=6)
        ax1.plot(m_vals, best_times, 's-', label='Best (Custom)', linewidth=2, markersize=6)
        ax1.plot(m_vals, reg_times, '^-', label='Registered', linewidth=2, markersize=6)
        ax1.set_xlabel('M', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title(f'Execution Time vs M\n(N={n}, K={k})', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup vs M
        ax2 = axes[1]
        speedup_best = ref_times / best_times
        speedup_reg = ref_times / reg_times
        ax2.plot(m_vals, speedup_best, 's-', label='Best vs Reference', linewidth=2, markersize=6, color='green')
        ax2.plot(m_vals, speedup_reg, '^-', label='Registered vs Reference', linewidth=2, markersize=6, color='orange')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
        ax2.set_xlabel('M', fontsize=12)
        ax2.set_ylabel('Speedup (x)', fontsize=12)
        ax2.set_title(f'Speedup vs M\n(N={n}, K={k})', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"benchmark_n{n}_k{k}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fig_path}")
    
    # Generate a summary heatmap figure
    print("\n=== Generating Summary Heatmap ===")
    generate_heatmap(results, m_values, n_values, output_dir)
    
    # Generate speedup plot with all (m, n, k) tuples sorted by m*n*k
    print("\n=== Generating Speedup vs Problem Size Plot ===")
    generate_speedup_vs_problem_size(results, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}/")


def generate_speedup_vs_problem_size(results, output_dir):
    """Generate a plot with all (m, n, k) tuples on x-axis sorted by m*n*k, speedup on y-axis."""
    # Collect all data points
    all_data = []
    for (m, n), data in results.items():
        for i, k in enumerate(data['k']):
            problem_size = m * n * k
            ref_time = data['reference'][i]
            best_time = data['best'][i]
            reg_time = data['registered'][i]
            speedup_best = ref_time / best_time
            speedup_reg = ref_time / reg_time
            all_data.append({
                'm': m, 'n': n, 'k': k,
                'problem_size': problem_size,
                'speedup_best': speedup_best,
                'speedup_reg': speedup_reg,
                'ref_time': ref_time,
                'best_time': best_time,
                'reg_time': reg_time,
            })
    
    # Sort by problem size (m * n * k)
    all_data.sort(key=lambda x: x['problem_size'])
    
    # Create x-axis labels and values
    x_labels = [f"({d['m']},{d['n']},{d['k']})" for d in all_data]
    x_indices = np.arange(len(all_data))
    speedup_best = [d['speedup_best'] for d in all_data]
    speedup_reg = [d['speedup_reg'] for d in all_data]
    problem_sizes = [d['problem_size'] for d in all_data]
    
    # Figure 1: Speedup vs problem size with (m,n,k) labels
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(x_indices, speedup_best, 's-', label='Best vs Reference', 
            linewidth=1.5, markersize=4, color='green', alpha=0.8)
    ax.plot(x_indices, speedup_reg, '^-', label='Registered vs Reference', 
            linewidth=1.5, markersize=4, color='orange', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
    
    ax.set_xlabel('(M, N, K) tuples sorted by M×N×K', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Speedup vs Problem Size (all configurations)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks - show every Nth label to avoid overcrowding
    n_labels = len(x_labels)
    if n_labels > 30:
        step = max(1, n_labels // 20)
        ax.set_xticks(x_indices[::step])
        ax.set_xticklabels(x_labels[::step], rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "speedup_vs_problem_size.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")
    
    # Figure 2: Speedup vs m*n*k (continuous x-axis)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(problem_sizes, speedup_best, s=40, label='Best vs Reference', 
               color='green', alpha=0.7, marker='s')
    ax.scatter(problem_sizes, speedup_reg, s=40, label='Registered vs Reference', 
               color='orange', alpha=0.7, marker='^')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
    
    ax.set_xlabel('Problem Size (M × N × K)', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Speedup vs Problem Size', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "speedup_vs_mnk_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def generate_heatmap(results, m_values, n_values, output_dir):
    """Generate heatmaps showing average speedup for each (m, n) pair."""
    # Calculate average speedup for each (m, n)
    avg_speedups = np.zeros((len(m_values), len(n_values)))
    
    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            if (m, n) in results and len(results[(m, n)]['k']) > 0:
                ref_times = np.array(results[(m, n)]['reference'])
                best_times = np.array(results[(m, n)]['best'])
                avg_speedups[i, j] = np.mean(ref_times / best_times)
            else:
                avg_speedups[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(avg_speedups, cmap='RdYlGn', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(n_values)))
    ax.set_yticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yticklabels([str(m) for m in m_values])
    
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('M', fontsize=12)
    ax.set_title('Average Speedup (Best vs Reference)\nAveraged across K values', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (x)', fontsize=12)
    
    # Annotate cells with values (only if not too many cells)
    if len(m_values) * len(n_values) <= 100:
        for i in range(len(m_values)):
            for j in range(len(n_values)):
                if not np.isnan(avg_speedups[i, j]):
                    text = ax.text(j, i, f'{avg_speedups[i, j]:.2f}x',
                                  ha='center', va='center', color='black', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "summary_heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


def save_results_to_csv(results, output_dir):
    """Save benchmark results to CSV file."""
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'n', 'k', 'reference_time_ms', 'best_time_ms', 'registered_time_ms', 'speedup_best', 'speedup_registered'])
        
        for (m, n), data in results.items():
            for i, k in enumerate(data['k']):
                ref_time = data['reference'][i]
                best_time = data['best'][i]
                reg_time = data['registered'][i]
                speedup_best = ref_time / best_time
                speedup_reg = ref_time / reg_time
                writer.writerow([m, n, k, ref_time, best_time, reg_time, speedup_best, speedup_reg])
    
    print(f"Saved results to: {csv_path}")



if __name__ == "__main__":
    main()
