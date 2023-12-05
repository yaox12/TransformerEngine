/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor fused_rope_forward(const at::Tensor &input, const at::Tensor &freqs,
                              const bool transpose_output_memory) {
  using namespace transformer_engine;
  TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(input.size(0) == freqs.size(0),
              "expected input and freqs tensor have the same sequence length");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(input.size(3) >= freqs.size(3),
              "expected the last dim of the input tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  // input sizes: (s, b, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s = input.size(0);
  const int b = input.size(1);
  const int h = input.size(2);
  const int d = input.size(3);
  // input strides
  const int stride_s = input.stride(0);
  const int stride_b = input.stride(1);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = freqs.size(3);

  // output
  auto act_options = input.options().requires_grad(false);
  at::Tensor output;
  if (transpose_output_memory) {
    output = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    output = torch::empty({s, b, h, d}, act_options);
  }
  // output strides
  const int o_stride_s = output.stride(0);
  const int o_stride_b = output.stride(1);
  const int o_stride_h = output.stride(2);
  const int o_stride_d = output.stride(3);

  auto input_cu = makeTransformerEngineTensor(input);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto output_cu = makeTransformerEngineTensor(output);

  nvte_fused_rope_forward(input_cu.data(), freqs_cu.data(), output_cu.data(), s,
                          b, h, d, d2, stride_s, stride_b, stride_h, stride_d,
                          o_stride_s, o_stride_b, o_stride_h, o_stride_d,
                          at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor fused_rope_backward(const at::Tensor &output_grads,
                               const at::Tensor &freqs,
                               const bool transpose_output_memory) {
  using namespace transformer_engine;
  TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(
      output_grads.size(0) == freqs.size(0),
      "expected output_grads and freqs tensor have the same sequence length");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(output_grads.size(3) >= freqs.size(3),
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  // output_grads sizes: (s, b, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s = output_grads.size(0);
  const int b = output_grads.size(1);
  const int h = output_grads.size(2);
  const int d = output_grads.size(3);
  // output_grads strides
  const int stride_s = output_grads.stride(0);
  const int stride_b = output_grads.stride(1);
  const int stride_h = output_grads.stride(2);
  const int stride_d = output_grads.stride(3);
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = freqs.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  at::Tensor input_grads;
  if (transpose_output_memory) {
    input_grads = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    input_grads = torch::empty({s, b, h, d}, act_options);
  }
  const int o_stride_s = input_grads.stride(0);
  const int o_stride_b = input_grads.stride(1);
  const int o_stride_h = input_grads.stride(2);
  const int o_stride_d = input_grads.stride(3);

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto input_grads_cu = makeTransformerEngineTensor(input_grads);

  nvte_fused_rope_backward(
      output_grads_cu.data(), freqs_cu.data(), input_grads_cu.data(), s, b, h, d,
      d2, stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
      o_stride_h, o_stride_d, at::cuda::getCurrentCUDAStream());

  return input_grads;
}
