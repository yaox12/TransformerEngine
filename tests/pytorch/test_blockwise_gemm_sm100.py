# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import torch

from transformer_engine.pytorch.cpp_extensions.blockwise_gemm_sm100 import (
    blockwise_gemm_sm100,
    blockwise_grouped_gemm_sm100,
)
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm, general_gemm
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
import transformer_engine_torch as tex


@pytest.mark.parametrize("num_gemms", [4])
@pytest.mark.parametrize("m", [2048])
@pytest.mark.parametrize("k", [7168])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("layout", ["TN", "NN"])
def test_blockwise_grouped_gemm_sm100(layout, num_gemms, m, k, n):
    quantizer_w = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )
    quantizer_x = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
    )

    m_splits = [m] * num_gemms
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    if layout == "TN":
        weights = [
            quantizer_w.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
            for _ in range(num_gemms)
        ]
        activations = [
            quantizer_x.quantize(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            for m in m_splits
        ]

        out_ref = torch.empty(sum(m_splits), n, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(sum(m_splits), n, dtype=torch.bfloat16, device="cuda")
    elif layout == "NN":
        weights = [
            quantizer_w.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
            for _ in range(num_gemms)
        ]
        activations = [
            quantizer_x.quantize(torch.randn(m, n, dtype=torch.bfloat16, device="cuda"))
            for m in m_splits
        ]

        out_ref = torch.empty(sum(m_splits), k, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(sum(m_splits), k, dtype=torch.bfloat16, device="cuda")

    blockwise_grouped_gemm_sm100(
        weights, transa, activations, transb, [out], tex.DType.kBFloat16, m_splits, accumulate=False
    )

    general_grouped_gemm(
        weights,
        activations,
        [out_ref],
        [None] * num_gemms,
        torch.bfloat16,
        layout=layout,
        m_splits=m_splits,
        single_output=True,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
