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

#include "core/ps/optimizer/optimizer_kernel.h"

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

    void Serialized(butil::IOBuf& buf) const;

    void DeSerialized(butil::IOBuf& buf);

private:
    float beta1_power_ = 0;
    float beta2_power_ = 0;

    Eigen::ArrayXf w_;
    Eigen::ArrayXf m_;
    Eigen::ArrayXf v_;
};

typedef DenseKernelBlock<Adam, DenseAdamValue> DenseAdamKernelBlock;

class SparseAdamValue {
public:
    SparseAdamValue(int dim, const Adam* opt);

    ~SparseAdamValue() {
        if (!IsMiniDim()) {
            delete w_.p;
            delete m_.p;
            delete v_.p;
        }
    }

    int Dim() { return dim_; }

    bool IsMiniDim() {
        // UnionWeight could store two float
        if (2 > dim_) {
            return true;
        } else {
            return false;
        }
    }

    float* Weight() {
        if (IsMiniDim()) {
            return w_.v;
        } else {
            return w_.p;
        }
    }

    uint32_t Version() {
        return version_;
    }

    void IncreaseVersion() {
        version_++;
    }

    void AddShow(int show) {
        show_ += show;
    }

    void Apply(const Adam* opt, SparseGradInfo& grad_info);

    void Serialized(butil::IOBuf& buf);

    void DeSerialized(butil::IOBuf& buf);

protected:
    float* M() {
        if (IsMiniDim()) {
            return m_.v;
        } else {
            return m_.p;
        }
    }

    float* V() {
        if (IsMiniDim()) {
            return v_.v;
        } else {
            return v_.p;
        }
    }

private:
    UnionWeight w_;
    UnionWeight m_;
    UnionWeight v_;
    int dim_ = 0;
    uint32_t version_ = 0;
    int show_ = 0;
};

typedef SparseKernelBlock<Adam, SparseAdamValue> SparseAdamKernelBlock;

} // namespace tensornet {

#endif // !TENSORNET_OPTIMIZER_ADAM_KERNEL_H_
