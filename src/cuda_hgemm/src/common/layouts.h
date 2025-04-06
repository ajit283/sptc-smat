#pragma once

#include "cute/algorithm/fill.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/tensor_view.h"
#include <curand_kernel.h>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cute;

// Define layout and matrix layout globally or in main
auto TCBlocksSmall = make_layout(make_shape(Int<16>{}, Int<16>{}));
auto TCBlocksSmall_shape = make_shape(Int<16>{}, Int<16>{});
auto outer = make_layout(make_shape((size_t)2048, (size_t)2048));

auto TCBlocksLarge = make_layout(make_shape(Int<16>{}, Int<32>{}));
auto TCBlocksLarge_shape = make_shape(Int<16>{}, Int<32>{});

auto matrix_blocks = blocked_product(TCBlocksSmall, outer);

auto matrix_blocks_large = blocked_product(TCBlocksLarge, outer);
