# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""BackwardCast API"""
from typing import Union, Optional, Tuple, List, Dict, Any

import torch

from .. import cpp_extensions as tex

from .base import TransformerEngineBaseModule

from ..cpp_extensions import cast_to_fp8
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..graph import is_graph_capturing
from ..jit import no_torch_dynamo
from ..float8_tensor import Float8Tensor
from ..utils import requires_grad

__all__ = ["BackwardCast"]


class _BackwardCast(torch.autograd.Function):
    """functional BackwardCast
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        is_grad_enabled: bool,
        activation_dtype: torch.dtype,
        fp8_meta: Dict[str, Any],
    ) -> torch.Tensor:

        if is_grad_enabled:
            ctx.activation_dtype = activation_dtype
            ctx.fp8_meta = fp8_meta
            ctx.inp_shape = inp.shape
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if requires_grad(inp,):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or FP8GlobalStateManager.is_first_fp8_module()
                )

        return inp

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        with torch.cuda.nvtx.range("_BackwardCast"):
            fp8_dtype_backward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )
            grad_output = cast_to_fp8(
                grad_output,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
            dgrad = Float8Tensor(
                data=grad_output.view(ctx.inp_shape),
                fp8_meta=ctx.fp8_meta,
                fp8_meta_forward=False,
                fp8_meta_index=tex.FP8BwdTensors.GRAD_OUTPUT1,
                dtype=ctx.activation_dtype,
            )

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dgrad,
            None,
            None,
            None,
        )


class BackwardCast(TransformerEngineBaseModule):
    r"""
    Forward no-op and backward cast to fp8.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        return [None, None]

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """
        BackwardCast FWD.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        """
        with self.prepare_forward(inp, None) as inp:
            assert self.fp8, "Need to run inside fp8_autocast region."
            self.set_activation_dtype(inp)

            if torch.is_grad_enabled():
                fwd_fn = _BackwardCast.apply
                args = []
            else:
                fwd_fn = _BackwardCast.forward
                args = [None]

            args += (
                inp,
                torch.is_grad_enabled(),
                self.activation_dtype,
                self.fp8_meta,
            )

            return fwd_fn(*args)
