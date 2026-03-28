"""
Debug: check epilogue with different batch sizes on Modal B200.
"""

import modal

app = modal.App("test-cutedsl-debug")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "nvidia-cutlass-dsl")
)


@app.function(image=image, gpu="B200:1", timeout=600)
def test_debug():
    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.tcgen05 as tcgen05
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass import Float32, BFloat16
    from cutlass.cute.runtime import from_dlpack

    M, N = 64, 64
    K_CKV = 512

    # Test with different batch sizes
    for batch in [32, 64, 96, 128, 1, 2, 8]:
        # Create output tensor
        S_base = torch.zeros(batch, M, N, dtype=torch.float32, device="cuda")
        S_3d = S_base.permute(1, 2, 0)  # [M, N, batch]

        mS = from_dlpack(S_3d, assumed_align=16)

        c_layout = utils.LayoutEnum.from_tensor(mS)
        print(f"batch={batch}: shape={mS.shape}, leading_dim={mS.leading_dim}, c_layout={c_layout}, is_m_major={c_layout.is_m_major_c()}")

        # Try the epi_tile computation
        cta_group = tcgen05.CtaGroup.ONE

        # Create a simple MMA
        A_base = torch.zeros(batch, M, K_CKV, dtype=torch.bfloat16, device="cuda")
        A_3d = A_base.permute(1, 2, 0)
        mA = from_dlpack(A_3d, assumed_align=16)

        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = a_major  # same layout for B

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16, a_major, b_major, Float32, cta_group, (M, N),
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        use_2cta_instrs = atom_thr_size == 2
        k_tile = cute.size(tiled_mma.shape_mnk, mode=[2]) * 4
        cta_tile_shape_mnk = (M // atom_thr_size, N, k_tile)
        epi_tile = cta_tile_shape_mnk[:2]

        print(f"  atom_thr_size={atom_thr_size}, use_2cta={use_2cta_instrs}")
        print(f"  cta_tile={cta_tile_shape_mnk}, epi_tile={epi_tile}")

        # Check: tmem_warp_shape
        if cta_tile_shape_mnk[0] == 64 and use_2cta_instrs:
            tmem_warp = (2, 2)
        else:
            tmem_warp = (4, 1)

        epi_warp = (epi_tile[0] // tmem_warp[0], epi_tile[1] // tmem_warp[1])
        num_dp = epi_warp[0]
        print(f"  tmem_warp={tmem_warp}, epi_warp={epi_warp}, num_dp={num_dp}")

        try:
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                cta_tile_shape_mnk, c_layout, mS.element_type,
                Float32, epi_tile, use_2cta_instrs,
            )
            print(f"  get_tmem_load_op: OK")
        except Exception as e:
            print(f"  get_tmem_load_op: FAILED - {e}")


@app.local_entrypoint()
def main():
    test_debug.remote()
