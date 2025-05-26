// Copyright 2020-2025 Qihoo Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_
#define TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {

class BalanceDatasetOp : public tensorflow::UnaryDatasetOpKernel {
public:
    static constexpr const char* const kDatasetType = "Balance";
    static constexpr const char* const kInputDataset = "input_dataset";
    static constexpr const char* const kOutputTypes = "output_types";
    static constexpr const char* const kOutputShapes = "output_shapes";

    explicit BalanceDatasetOp(tensorflow::OpKernelConstruction* ctx);

protected:
    void MakeDataset(tensorflow::OpKernelContext* ctx,
                     tensorflow::DatasetBase* input,
                     tensorflow::DatasetBase** output) override;

private:
    class Dataset;
};

}  // namespace tensorflow

#endif  // TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_
