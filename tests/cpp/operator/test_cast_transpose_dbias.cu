/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename IT, typename OT, typename CT>
void compute_ref_cast_transpose_dbias(const IT *input_h,
                                      const CT scale,
                                      OT *output_c_h,
                                      OT *output_t_h,
                                      CT *amax_h,
                                      IT *dbias_h,
                                      const size_t N,
                                      const size_t H) {
  CT amax  = 0.;

  std::vector<CT> acc_dbias(H, 0.);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);

      // update amax
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;

      output_c_h[i * H + j] = static_cast<OT>(scale * elt);
      output_t_h[j * N + i] = static_cast<OT>(scale * elt);

      // dbias
      acc_dbias[j] += elt;
    }
  }

  *amax_h = amax;

  for (size_t i = 0; i < H; i++) {
    dbias_h[i] = static_cast<IT>(acc_dbias[i]);
  }
}

template <typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;
  using CType = fp32;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input("input", std::vector<size_t>{N, H}, itype);

  Tensor output("output", std::vector<size_t>{N, H}, otype, true, true);
  // dbias has the same data type with "output grad"
  Tensor dbias("dbias", std::vector<size_t>{H}, itype);

  fillUniform(&input);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output_c = std::make_unique<OType[]>(N*H);
  std::unique_ptr<OType[]> ref_output_t = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_output_dbias = std::make_unique<IType[]>(H);

  CType ref_amax;
  compute_ref_cast_transpose_dbias(input.rowwise_cpu_dptr<IType>(),
                                   output.scale(),
                                   ref_output_c.get(),
                                   ref_output_t.get(),
                                   &ref_amax,
                                   ref_output_dbias.get(),
                                   N, H);

  Tensor workspace;

  nvte_quantize_dbias(input.data(),
                      output.data(),
                      dbias.data(),
                      workspace.data(),
                      0);

  workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());


  nvte_quantize_dbias(input.data(),
                      output.data(),
                      dbias.data(),
                      workspace.data(),
                      0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / output.scale();
    compareResults("scale_inv", output.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output, ref_output_c.get(), true, atol, rtol);
  compareResults("output_t", output, ref_output_t.get(), false, atol, rtol);

  auto [atol_dbias, rtol_dbias] = getTolerances(itype);
  rtol_dbias *= 4;
  compareResults("output_dbias", dbias, ref_output_dbias.get(), true, atol_dbias, rtol_dbias);
}

std::vector<std::pair<size_t, size_t>> test_cases = {{64, 400},
                                                     {2048, 12288},
                                                     {768, 1024},
                                                     {256, 65536},
                                                     {65536, 128},
                                                     {256, 256}};

}  // namespace;


class CTDBiasTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                 transformer_engine::DType,
                                                                 std::pair<size_t, size_t>>> {};

TEST_P(CTDBiasTestSuite, TestCTDBias) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<InputType, OutputType>(size.first, size.second);
      );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CTDBiasTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<CTDBiasTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
