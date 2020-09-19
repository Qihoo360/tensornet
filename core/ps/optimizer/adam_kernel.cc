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

#include "core/ps/optimizer/adam_kernel.h"

#include <butil/logging.h>

#include "core/utility/random.h"

using namespace Eigen;

namespace tensornet {

DenseAdamValue::DenseAdamValue(const Adam* opt, int len) {
    beta1_power_ = 1;
    beta2_power_ = 1;

    // NOTE, w must initialized as random. You can also call SetWeight initialize.
    // eigen init uniformly spread through [-1:1] default.
    w_.setRandom(len);
    w_ *= opt->initial_scale;

    // NOTE, m and v must be initialized zero
    m_.setZero(len);
    v_.setZero(len);
}

void DenseAdamValue::SetWeight(butil::IOBuf& w_buf) {
    CHECK_EQ(w_.size() * sizeof(float), w_buf.size());

    w_buf.copy_to(w_.data(), w_.size() * sizeof(float));
}

void DenseAdamValue::Apply(const Adam* opt, const Eigen::ArrayXf& g) {
    beta1_power_ *= opt->beta1;
    beta2_power_ *= opt->beta2;

    const float alpha = opt->learning_rate
        * Eigen::numext::sqrt(1.0 - beta2_power_) / (1.0 - beta1_power_);

    m_ += (g - m_) * (1.0 - opt->beta1);
    v_ += (g.square() - v_) * (1.0 - opt->beta2);
    w_ -= (m_ * alpha) / (v_.sqrt() + opt->epsilon);
}

std::ostream& operator<<(std::ostream& os, const DenseAdamValue& value) {
    int array_size = value.w_.size();

    os << "array_size:" << array_size << std::endl;
    os << "beta1_power:" << value.beta1_power_ << std::endl;
    os << "beta2_power:" << value.beta2_power_ << std::endl;

    for (int i = 0; i < array_size; i++) {
        os << value.w_[i] << "\t"
           << value.m_[i] << "\t"
           << value.v_[i] << std::endl;
    }

    return os;
}

std::istream& operator>>(std::istream& is, DenseAdamValue& value) {
    int array_size = 0;
    is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> array_size;

    CHECK_EQ(array_size, value.w_.size());

    is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> value.beta1_power_;
    is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> value.beta2_power_;

    for (int i = 0; i < array_size; i++) {
        is >> value.w_[i];
        is >> value.m_[i];
        is >> value.v_[i];
    }

    return is;
}

SparseAdamValue::SparseAdamValue(int dim, const Adam* opt) {
    dim_ = dim;

    if (!IsMiniDim_()) {
        w_.p = data + dim * 0;
        m_.p = data + dim * 1;
        v_.p = data + dim * 2;
    }

    auto& reng = local_random_engine();
    auto distribution = std::normal_distribution<float>(0, 1 / sqrt(Dim()));

    if (IsMiniDim_()) {
        for (int i = 0; i < Dim(); ++i) {
            w_.v[i] = distribution(reng) * opt->initial_scale;
            m_.v[i] = 0;
            v_.v[i] = 0;
        }
    } else {
        for (int i = 0; i < Dim(); ++i) {
            w_.p[i] = distribution(reng) * opt->initial_scale;
            m_.p[i] = 0;
            v_.p[i] = 0;
        }
    }
}

void SparseAdamValue::Apply(const Adam* opt, SparseGradInfo& grad_info) {
    show_ += grad_info.batch_show;

    float* w = Weight();
    float* m = M();
    float* v = V();

    for (int i = 0; i < dim_; ++i) {
        m[i] = opt->beta1 * m[i] + (1 - opt->beta1) * grad_info.grad[i];
        v[i] = opt->beta2 * v[i] + (1 - opt->beta2) * grad_info.grad[i] * grad_info.grad[i];

        w[i] -= opt->learning_rate * m[i] / (opt->epsilon + sqrt(v[i]));
    }
}

std::ostream& operator<<(std::ostream& os, const SparseAdamValue& value) {
    os << value.dim_ << "\t";

    for (int i = 0; i < value.dim_; i++) {
        os << value.Weight()[i] << "\t";
        os << value.M()[i] << "\t";
        os << value.V()[i] << "\t";
    }

    os << value.show_;

    return os;
}

std::istream& operator>>(std::istream& is, SparseAdamValue& value) {
    int dim;
    is >> dim;

    CHECK_EQ(dim, value.dim_);

    for (int i = 0; i < value.dim_; i++) {
        is >> value.Weight()[i];
        is >> value.M()[i];
        is >> value.V()[i];
    }

    is >> value.show_;

    return is;
}

} // namespace tensornet {

