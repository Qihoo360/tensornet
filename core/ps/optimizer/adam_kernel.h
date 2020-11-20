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

#ifndef TENSORNET_OPTIMIZER_ADAM_KERNEL_H_
#define TENSORNET_OPTIMIZER_ADAM_KERNEL_H_

#include "core/ps/optimizer/optimizer.h"

#include <butil/iobuf.h>
#include <Eigen/Dense>

#include "core/ps/optimizer/data_struct.h"

namespace tensornet {

class DenseAdamValue {
public:
    DenseAdamValue(const Adam* opt, int len);

    void SetWeight(butil::IOBuf& w_buf);

    const Eigen::ArrayXf& GetWeight() const {
        return w_;
    }

    void Apply(const Adam* opt, const Eigen::ArrayXf& g);

    size_t DataSize() const {
        return sizeof(float) * 2
            + (m_.size() + v_.size() + w_.size()) * sizeof(float);
    }

    friend std::ostream& operator<<(std::ostream& os, const DenseAdamValue& value);
    friend std::istream& operator>>(std::istream& is, DenseAdamValue& value);

private:
    float beta1_power_ = 0;
    float beta2_power_ = 0;

    Eigen::ArrayXf w_;
    Eigen::ArrayXf m_;
    Eigen::ArrayXf v_;
};

std::ostream& operator<<(std::ostream& os, const DenseAdamValue& value);
std::istream& operator>>(std::istream& is, DenseAdamValue& value);

struct alignas(4) SparseAdamValue {
public:
    SparseAdamValue(int dim, const Adam* opt);
    ~SparseAdamValue() = default;

    static constexpr int DynSizeof(int dim) {
        return sizeof(SparseAdamValue) + sizeof(float) * dim * 3;
    }

    int Dim() const {
        return dim_;
    }

    float* Weight() {
        return data_ + Dim() * 0;
    }

    const float* Weight() const {
        return data_ + Dim() * 0;
    }

    void Apply(const Adam* opt, SparseGradInfo& grad_info);

    void ShowDecay(const Adam* opt) {}

protected:

    float* M() {
        return data_ + Dim() * 1;
    }

    const float* M() const {
        return data_ + Dim() * 1;
    }

    float* V() {
        return data_ + Dim() * 2;
    }

    const float* V() const {
        return data_ + Dim() * 2;
    }

    friend std::ostream& operator<<(std::ostream& os, const SparseAdamValue& value);
    friend std::istream& operator>>(std::istream& is, SparseAdamValue& value);

private:
    int dim_ = 0;
    float show_ = 0.0;
    float data_[0];
};

std::ostream& operator<<(std::ostream& os, const SparseAdamValue& value);
std::istream& operator>>(std::istream& is, SparseAdamValue& value);

} // namespace tensornet {

#endif // !TENSORNET_OPTIMIZER_ADAM_KERNEL_H_
