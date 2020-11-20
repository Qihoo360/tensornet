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

#ifndef TENSORNET_OPTIMIZER_FTRL_KERNEL_H_
#define TENSORNET_OPTIMIZER_FTRL_KERNEL_H_

#include "core/ps/optimizer/optimizer.h"

#include <butil/iobuf.h>
#include <Eigen/Dense>

#include "core/ps/optimizer/data_struct.h"

namespace tensornet {

class DenseFtrlValue {
public:
    DenseFtrlValue(const Ftrl* opt, int len);

    void SetWeight(butil::IOBuf& w_buf);

    const Eigen::ArrayXf& GetWeight() const {
        return w_;
    }

    void Apply(const Ftrl* opt, const Eigen::ArrayXf& g);

    size_t DataSize() const {
        return w_.size() * sizeof(float) * 4;
    }

    friend std::ostream& operator<<(std::ostream& os, const DenseFtrlValue& value);
    friend std::istream& operator>>(std::istream& is, DenseFtrlValue& value);

private:
    Eigen::ArrayXf w_;
    Eigen::ArrayXf z_;
    Eigen::ArrayXf n_;
};

std::ostream& operator<<(std::ostream& os, const DenseFtrlValue& value);
std::istream& operator>>(std::istream& is, DenseFtrlValue& value);

struct alignas(4) SparseFtrlValue {
public:
    SparseFtrlValue(int dim, const Ftrl* opt);

    ~SparseFtrlValue() = default;

    static constexpr int DynSizeof(int dim) {
        return sizeof(SparseFtrlValue) + sizeof(float) * dim * 3;
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

    void Apply(const Ftrl* opt, SparseGradInfo& grad_info);

    void ShowDecay(const Ftrl* opt);

    friend std::ostream& operator<<(std::ostream& os, const SparseFtrlValue& value);
    friend std::istream& operator>>(std::istream& is, SparseFtrlValue& value);

protected:
    float* Z() {
        return data_ + Dim() * 1;
    }

    const float* Z() const {
        return data_ + Dim() * 1;
    }

    float* N() {
        return data_ + Dim() * 2;
    }

    const float* N() const {
        return data_ + Dim() * 2;
    }

private:
    int dim_ = 0;
    float show_ = 0.0;
    float data_[0];
};

std::ostream& operator<<(std::ostream& os, const SparseFtrlValue& value);
std::istream& operator>>(std::istream& is, SparseFtrlValue& value);

}  // namespace tensornet

#endif  // !TENSORNET_OPTIMIZER_FTRL_KERNEL_H_
