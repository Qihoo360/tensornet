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
#include <iostream>
#include <sstream>

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

    virtual std::tuple<bool, std::string> NeedOldCompat(std::istream& is, int dim) const {
        std::string emptyString = "";
        return std::make_tuple(false, emptyString);
    }

public:
    float learning_rate = 0.01;
    float show_decay_rate = 0.98;
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

    std::tuple<bool, std::string> NeedOldCompat(std::istream& is, int dim) const {
        bool need_old_compat = false;
        std::string line;
        std::string cell;
        std::getline(is, line); // 抹去换行符
        std::getline(is, line);
        std::istringstream iss(line);
        int column_count = 0;

        while (std::getline(iss, cell, '\t')) {
            ++column_count;
        }

        // columns should be sign, dim_, dims_ * weight, g2sum, show, no_show_days
        // if columnCount is 12,  means no no_show_days column
        if(column_count == dim + 4){
            need_old_compat = true;
        }

        return std::make_tuple(need_old_compat, line);
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
