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

#ifndef TENSORNET_OPTIMIZER_OPTIMIZER_H_
#define TENSORNET_OPTIMIZER_OPTIMIZER_H_

#include <vector>
#include <memory>
#include <string>

namespace tensornet {

class DenseOptimizerKernelBase;
class SparseOptimizerKernelBase;

typedef std::shared_ptr<DenseOptimizerKernelBase> DenseOptKernelSharedPtr;
typedef std::shared_ptr<SparseOptimizerKernelBase> SparseOptKernelSharedPtr;

class OptimizerBase {
public:
    virtual DenseOptKernelSharedPtr CreateDenseOptKernel(
        int offset_begin, int offset_end) const = 0;

    virtual SparseOptKernelSharedPtr CreateSparseOptKernel(int dimension) const = 0;

    virtual std::string Name() const = 0;

public:
    float learning_rate = 0.01;
    float show_decay_rate = 0.98;
    float feature_drop_show = 0.02;
};

class Adam : public OptimizerBase {
public:
    virtual DenseOptKernelSharedPtr CreateDenseOptKernel(
        int offset_begin, int offset_end) const;

    virtual SparseOptKernelSharedPtr CreateSparseOptKernel(int dimension) const;

    virtual std::string Name() const {
        return "Adam";
    }

public:
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-08;
    float initial_scale = 1.0;
};

class AdaGrad : public OptimizerBase {
public:
    virtual DenseOptKernelSharedPtr CreateDenseOptKernel(
        int offset_begin, int offset_end) const;

    virtual SparseOptKernelSharedPtr CreateSparseOptKernel(int dimension) const;

    virtual std::string Name() const {
        return "AdaGrad";
    }

public:
    float initial_g2sum = 0;
    float initial_scale = 1.0;
    float epsilon = 1e-08;
    float grad_decay_rate = 1.0;
    float mom_decay_rate = 1.0;
    float show_decay_rate = 1.0;
    float show_threshold = 0.0;
    int no_show_days = 1000;
};

class Ftrl : public OptimizerBase {
public:
    virtual DenseOptKernelSharedPtr CreateDenseOptKernel(
        int offset_begin, int offset_end) const;

    virtual SparseOptKernelSharedPtr CreateSparseOptKernel(int dimension) const;

    virtual std::string Name() const {
        return "Ftrl";
    }

public:
    float initial_scale = 0;
    float beta = 1;
    float lambda1 = 0.1;
    float lambda2 = 1;
    float show_decay_rate = 1.0;
    float show_threshold = 0.0;
};

} // namespace tensornet {

#endif // !TENSORNET_OPTIMIZER_OPTIMIZER_H_
