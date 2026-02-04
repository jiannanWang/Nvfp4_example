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


def main():
    seed = 1111
    output_dir = "benchmark_figures_deepseek"
    shapes_file = "deepseek_shapes.csv"
    os.makedirs(output_dir, exist_ok=True)

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
    # Aggregate across all (m, n) pairs
    all_correct_best = []
    all_registered = []
    all_reference = []
    for data in results.values():
        all_correct_best.extend(data['correct_best'])
        all_registered.extend(data['registered'])
        all_reference.extend(data['reference'])
    # How many registered kernels are correct?
    print(f"Correct best: {sum(all_correct_best)}/{len(all_correct_best)}")
    # how many registered kernels are faster than ref?
    faster_count = sum(r < ref for r, ref in zip(all_registered, all_reference))
    print(f"Registered faster than ref: {faster_count}/{len(all_registered)}")
    
    # Generate figures for each (m, n) pair
    print(f"\n=== Generating Figures ===")
    for (m, n), data in results.items():
        if len(data['k']) == 0:
            continue
            
        # Sort by k for proper
    
    # Generate figures for each (m, n) pair
    print(f"\n=== Generating Figures ===")
    for (m, n), data in results.items():
        if len(data['k']) == 0:
            continue
            
        # Sort by k for proper line plots
        sorted_indices = np.argsort(data)

    
    # Generate figures for each (m, n) pair
    print(f"\n=== Generating Figures ===")
    for (m, n), data in results.items():
        if len(data['k']) == 0:
            continue
    
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}/{len(args_dict)}")
    print(f"Failed: {failed}/{len(args_dict)}")
    
    # Generate figures for each (m, n) pair
    print(f"\n=== Generating Figures ===")
    for (m, n), data in results.items():
        if len(data['k']) == 0:
            continue
            
        # Sort by k for proper line plots
        sorted_indices = np.argsort(data['k'])
        k_values = np.array(data['k'])[sorted_indices]
        ref_times = np.array(data['reference'])[sorted_indices]
        best_times = np.array(data['best'])[sorted_indices]
        reg_times = np.array(data['registered'])[sorted_indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Execution time vs K
        ax1 = axes[0]
        ax1.plot(k_values, ref_times, 'o-', label='Reference', linewidth=2, markersize=6)
        ax1.plot(k_values, best_times, 's-', label='Best (Custom)', linewidth=2, markersize=6)
        ax1.plot(k_values, reg_times, '^-', label='Registered', linewidth=2, markersize=6)
        ax1.set_xlabel('K', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title(f'Execution Time vs K\n(M={m}, N={n})', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup vs K
        ax2 = axes[1]
        speedup_best = ref_times / best_times
        speedup_reg = ref_times / reg_times
        ax2.plot(k_values, speedup_best, 's-', label='Best vs Reference', linewidth=2, markersize=6, color='green')
        ax2.plot(k_values, speedup_reg, '^-', label='Registered vs Reference', linewidth=2, markersize=6, color='orange')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
        ax2.set_xlabel('K', fontsize=12)
        ax2.set_ylabel('Speedup (x)', fontsize=12)
        ax2.set_title(f'Speedup vs K\n(M={m}, N={n})', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"benchmark_m{m}_n{n}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fig_path}")
    
    # Generate a summary heatmap figure
    print("\n=== Generating Summary Heatmap ===")
    generate_heatmap(results, m_values, n_values, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}/")


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
