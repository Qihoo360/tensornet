// Copyright (c) 2020, Qihoo, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/ps/optimizer/optimizer_kernel.h"
#include "core/ps/optimizer/adam_kernel.h"
#include "core/ps/optimizer/ada_grad_kernel.h"
#include "core/ps/optimizer/ftrl_kernel.h"

#include <memory>

namespace tensornet {

typedef DenseKernelBlock<Adam, DenseAdamValue> DenseAdamKernelBlock;
typedef DenseKernelBlock<AdaGrad, DenseAdaGradValue> DenseAdaGradKernelBlock;
typedef DenseKernelBlock<Ftrl, DenseFtrlValue> DenseFtrlKernelBlock;

typedef SparseKernelBlock<Adam, SparseAdamValue> SparseAdamKernelBlock;
typedef SparseKernelBlock<AdaGrad, SparseAdaGradValue> SparseAdaGradKernelBlock;
typedef SparseKernelBlock<Ftrl, SparseFtrlValue> SparseFtrlKernelBlock;


DenseOptKernelSharedPtr Adam::CreateDenseOptKernel(
    int offset_begin, int offset_end) const {
    return std::make_shared<DenseOptimizerKernel<DenseAdamKernelBlock>>(
            this, offset_begin, offset_end);
}

SparseOptKernelSharedPtr Adam::CreateSparseOptKernel(int dimension) const {
    return std::make_shared<SparseOptimizerKernel<SparseAdamKernelBlock>>(this, dimension);
}

DenseOptKernelSharedPtr AdaGrad::CreateDenseOptKernel(
    int offset_begin, int offset_end) const {
    return std::make_shared<DenseOptimizerKernel<DenseAdaGradKernelBlock>>(
            this, offset_begin, offset_end);
}

SparseOptKernelSharedPtr AdaGrad::CreateSparseOptKernel(int dimension) const {
    return std::make_shared<SparseOptimizerKernel<SparseAdaGradKernelBlock>>(this, dimension);
}

DenseOptKernelSharedPtr Ftrl::CreateDenseOptKernel(
    int offset_begin, int offset_end) const {
    return std::make_shared<DenseOptimizerKernel<DenseFtrlKernelBlock>>(
            this, offset_begin, offset_end);
}

SparseOptKernelSharedPtr Ftrl::CreateSparseOptKernel(int dimension) const {
    return std::make_shared<SparseOptimizerKernel<SparseFtrlKernelBlock>>(this, dimension);
}

} // namespace tensornet {
