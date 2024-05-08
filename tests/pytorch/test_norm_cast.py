import torch
import torch.multiprocessing as mp
import transformer_engine.pytorch as te
import pytest


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_norm_cast(normalization):
    hidden_size = 256
    inp = torch.randn((16, hidden_size), dtype=torch.bfloat16, device="cuda")

    if normalization == "LayerNorm":
        norm_bf16 = te.LayerNorm(hidden_size, params_dtype=torch.bfloat16)
    elif normalization == "RMSNorm":
        norm_bf16 = te.RMSNorm(hidden_size, params_dtype=torch.bfloat16)
    norm_fp8 = te.NormCast(
        hidden_size, params_dtype=torch.bfloat16, normalization=normalization
    )

    with te.fp8_autocast():
        output_fp8 = norm_fp8(inp)
        output_bf16 = norm_bf16(inp)
    torch.testing.assert_close(output_fp8.bfloat16(), output_bf16, rtol=0.125, atol=0.0675)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_norm_cast_linear(tp_size, sequence_parallel, normalization):
    if tp_size == 1 and sequence_parallel:
        pytest.skip("sequence_parallel=True does nothing with tp_size=1")
    if tp_size > 1:
        mp.spawn(
            _test_norm_cast_linear,
            args=(tp_size, sequence_parallel, normalization),
            nprocs=tp_size,
        )
    else:
        _test_norm_cast_linear(0, tp_size, sequence_parallel, normalization)


def _test_norm_cast_linear(rank, tp_size, sequence_parallel, normalization):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = None
    if tp_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=tp_size,
            rank=rank,
            init_method="tcp://localhost:12345",
        )
        tp_group = torch.distributed.distributed_c10d._get_default_group()

    hidden_size = 256
    inp = torch.randn((16, hidden_size), dtype=torch.bfloat16, device=device)

    norm_fp8 = te.NormCast(
        hidden_size, params_dtype=torch.bfloat16, normalization=normalization
    )
    linear_fp8 = te.Linear(
        hidden_size,
        hidden_size,
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
        tp_size=tp_size,
        params_dtype=torch.bfloat16,
    )
    norm_linear = te.LayerNormLinear(
        hidden_size,
        hidden_size,
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
        tp_size=tp_size,
        params_dtype=torch.bfloat16,
        normalization=normalization,
    )

    with torch.no_grad():
        norm_linear.layer_norm_weight = torch.nn.Parameter(norm_fp8.weight.clone())
        if normalization != "RMSNorm":
            norm_linear.layer_norm_bias = torch.nn.Parameter(norm_fp8.bias.clone())
        norm_linear.weight = torch.nn.Parameter(linear_fp8.weight.clone())
        norm_linear.bias = torch.nn.Parameter(linear_fp8.bias.clone())

    with te.fp8_autocast():
        output_norm_cast_linear = linear_fp8(norm_fp8(inp))
        output_layernorm_linear = norm_linear(inp)

        output_norm_cast_linear.sum().backward()
        output_layernorm_linear.sum().backward()

    torch.testing.assert_close(output_norm_cast_linear, output_layernorm_linear)
    torch.testing.assert_close(norm_fp8.weight.grad, norm_linear.layer_norm_weight.grad)
    if normalization != "RMSNorm":
        torch.testing.assert_close(norm_fp8.bias.grad, norm_linear.layer_norm_bias.grad)
    torch.testing.assert_close(linear_fp8.weight.grad, norm_linear.weight.grad)
    torch.testing.assert_close(linear_fp8.bias.grad, norm_linear.bias.grad)

if __name__ == "__main__":
    # test_norm_cast()
    test_norm_cast_linear(2, True, "LayerNorm")
