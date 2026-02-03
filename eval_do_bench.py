import argparse
import torch
import torch.cuda
import torch.cuda.nvtx
import triton

from utils import set_seed
from reference import generate_input, to_blocked, ref_kernel
from best_submission import custom_kernel, compile_kernel

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


def main():
    seed = 1111

    set_seed(seed)
    _setup_registered_kernel(custom_kernel, compile_kernel)

    args_dict = (
        (128, 7168, 16384, 1),
        (128, 4096, 7168, 1),
        (128, 7168, 2048, 1),
    )
    for m, n, k, l in args_dict: 
        data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
        torch.cuda.synchronize()

        reference_time = triton.testing.do_bench(lambda: ref_kernel(data))
        torch.cuda.synchronize()

        best_time = triton.testing.do_bench(lambda: custom_kernel(data))
        torch.cuda.synchronize()

        registered_time = triton.testing.do_bench(lambda: torch.ops.nvfp4_bench._scaled_mm(*data))
        torch.cuda.synchronize()

        print(f"m={m}, n={n}, k={k}, l={l}")
        print(f"  Reference time: {reference_time}")
        print(f"  Best time: {best_time}")
        print(f"  Registered time: {registered_time}")
        print(f"  Registration overhead: {registered_time - best_time}")



if __name__ == "__main__":
    main()
