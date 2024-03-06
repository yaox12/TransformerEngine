# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NormCast API"""
import os
from typing import Union, Optional, Tuple, List, Dict, Any

import torch
from torch.nn import init

from .. import cpp_extensions as tex

from .base import (
    _prepare_backward,
    TransformerEngineBaseModule,
)
from ..utils import (
    cast_if_needed,
    assert_dim_for_fp8_exec,
)
from ..constants import dist_group_type
from ..jit import no_torch_dynamo
from ._common import _apply_normalization
from ..float8_tensor import Float8Tensor

__all__ = ["NormCast"]


class _NormCast(torch.autograd.Function):
    """functional NormCast
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        eps: float,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        is_grad_enabled: bool,
        activation_dtype: torch.dtype,
        fp8_meta: Dict[str, Any],
        normalization: str,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
    ) -> Float8Tensor:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        assert_dim_for_fp8_exec(inputmat)

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        ln_out_dtype = torch.uint8
        ln_out = torch.empty_like(inputmat, dtype=ln_out_dtype)
        ln_out, mu, rsigma = _apply_normalization(
            inputmat,
            ln_out,
            ln_weight,
            ln_bias,
            eps,
            True,
            fp8_meta,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
            is_grad_enabled,
        )

        if is_grad_enabled:
            ctx.save_for_backward(
                inputmat,
                ln_weight,
                mu,
                rsigma,
                # fp8_meta["scaling_fwd"].scale_inv.clone(),
            )
            ctx.activation_dtype = activation_dtype
            ctx.fp8_meta = fp8_meta
            ctx.inp_shape = inp.shape
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.normalization = normalization
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size

        return Float8Tensor(
            data=ln_out.view_as(inp),
            fp8_meta=fp8_meta,
            fp8_meta_index=tex.FP8FwdTensors.GEMM1_INPUT,
            dtype=activation_dtype,
        )

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            True, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_NormCast"
        ):
            inputmat, ln_weight, mu, rsigma = ctx.saved_tensors

            # LayerNorm/RMSNorm backward does not support FP8 input
            if isinstance(grad_output, Float8Tensor):
                grad_output = grad_output.from_float8(ctx.activation_dtype)

            grad_output = grad_output.contiguous()
            d_ln_out = grad_output.view(inputmat.shape)

            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    d_ln_out,
                    inputmat,
                    mu,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    d_ln_out,
                    inputmat,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dbeta = None

            return (
                dgrad.view(ctx.inp_shape),
                dgamma,
                dbeta,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


class NormCast(TransformerEngineBaseModule):
    r"""
    Applies layer normalization followed by casting to FP8.

    Parameters
    ----------
    hidden_size : int
                size of each input sample.
    eps : float, default = 1e-5
        a value added to the denominator of layer normalization for numerical stability.
    sequence_parallel : bool, default = `False`
                        if set to `True`, uses sequence parallelism.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                    it controls the type used to allocate the initial parameters. Useful when
                    the model is trained with lower precision and the original FP32 parameters
                    would not fit in GPU memory.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta

                         the RMSNorm formula changes to

                         .. math::
                            y = \frac{x}{RMS(x) + \varepsilon} * (1 + \gamma)
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        normalization: str = "LayerNorm",
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        params_dtype = (
            torch.get_default_dtype() if params_dtype is None else params_dtype
        )
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        assert normalization in [
            "LayerNorm",
            "RMSNorm",
        ], "Unsupported normalization type!"
        self.normalization = normalization
        self.weight = torch.nn.Parameter(
            torch.empty(
                hidden_size,
                device=device,
                dtype=params_dtype,
            )
        )
        if self.normalization == "LayerNorm":
            self.bias = torch.nn.Parameter(
                torch.empty(
                    hidden_size,
                    device=device,
                    dtype=params_dtype,
                )
            )
        else:
            self.bias = None
        self.sequence_parallel = sequence_parallel

        self.reset_parameters(defer_init=(device == "meta"))

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)
        if defer_init:
            return

        if self.weight.device == torch.device("meta"):
            self.weight = torch.nn.Parameter(
                torch.empty_like(self.weight, device="cuda")
            )
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        init.constant_(self.weight, float(not self.zero_centered_gamma))

        if self.bias is not None:
            if self.bias.device == torch.device("meta"):
                self.bias = torch.nn.Parameter(
                    torch.empty_like(self.bias, device="cuda")
                )
            setattr(self.bias, "sequence_parallel", self.sequence_parallel)
            init.zeros_(self.bias)

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        return [None, None]

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
    ) -> Float8Tensor:
        """
        NormCast FWD.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        """
        with self.prepare_forward(inp, None) as inp:
            assert self.fp8, "Need to run inside fp8_autocast region."
            self.set_activation_dtype(inp)

            if torch.is_grad_enabled():
                fwd_fn = _NormCast.apply
                args = []
            else:
                fwd_fn = _NormCast.forward
                args = [None]

            args += (
                inp,
                self.weight,
                self.bias,
                self.eps,
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                torch.is_grad_enabled(),
                self.activation_dtype,
                self.fp8_meta,
                self.normalization,
                self.tp_group,
                self.tp_size,
            )

            return fwd_fn(*args)
