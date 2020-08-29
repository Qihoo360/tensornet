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

void DenseAdamValue::Serialized(butil::IOBuf& buf) const {
    buf.append(&beta1_power_, sizeof(beta1_power_));
    buf.append(&beta2_power_, sizeof(beta2_power_));

    buf.append(m_.data(), m_.size() * sizeof(float));
    buf.append(v_.data(), v_.size() * sizeof(float));
    buf.append(w_.data(), w_.size() * sizeof(float));
}

void DenseAdamValue::DeSerialized(butil::IOBuf& buf) {
    CHECK_EQ(buf.size(), DataSize());

    buf.cutn(&beta1_power_, sizeof(beta1_power_));
    buf.cutn(&beta2_power_, sizeof(beta2_power_));

    buf.cutn(m_.data(), m_.size() * sizeof(float));
    buf.cutn(v_.data(), v_.size() * sizeof(float));
    buf.cutn(w_.data(), w_.size() * sizeof(float));
}

SparseAdamValue::SparseAdamValue(int dim, const Adam* opt) {
    dim_ = dim;

    if (!IsMiniDim_()) {
        float* buf = new float[dim * 3];
        w_.p = buf + dim * 0;
        m_.p = buf + dim * 1;
        v_.p = buf + dim * 2;
    }

    auto& reng = local_random_engine();
    auto distribution = std::normal_distribution<float>(0, 1 / sqrt(Dim()));

    for (int i = 0; i < Dim(); ++i) {
        if (IsMiniDim_()) {
            w_.v[i] = distribution(reng) * opt->initial_scale;
            m_.v[i] = 0;
            v_.v[i] = 0;
        } else {
            w_.p[i] = distribution(reng) * opt->initial_scale;
            m_.p[i] = 0;
            v_.p[i] = 0;
        }
    }
}

void SparseAdamValue::Apply(const Adam* opt, SparseGradInfo& grad_info) {
    CHECK_EQ(Dim(), grad_info.dim);

    version_++;
    show_ += grad_info.show;

    float* w = Weight();
    float* m = M();
    float* v = V();

    double g_scale = grad_info.show + opt->epsilon;
    if (grad_info.version < Version()) {
        g_scale *= sqrt(Version() - grad_info.version);
    }

    CHECK_EQ(dim_, grad_info.dim);

    for (int i = 0; i < dim_; ++i) {
        double scaled_grad = grad_info.grad[i] / g_scale;
        m[i] = opt->beta1 * m[i] + (1 - opt->beta1) * scaled_grad;
        v[i] = opt->beta2 * v[i] + (1 - opt->beta2) * scaled_grad * scaled_grad;

        w[i] -= opt->learning_rate * m[i] / (opt->epsilon + sqrt(v[i]));
    }
}

std::ostream& operator<<(std::ostream& os, const SparseAdamValue& value) {
    os << value.dim_ << "\t";

    for (int i = 0; i < value.dim_; i++) {
        os << value.Weight()[i] << "\t";
    }

    for (int i = 0; i < value.dim_; i++) {
        os << value.M()[i] << "\t";
    }

    for (int i = 0; i < value.dim_; i++) {
        os << value.V()[i] << "\t";
    }

    for (int i = 0; i < value.dim_; i++) {
        os << value.V()[i] << "\t";
    }

    os << value.version_ << "\t";
    os << value.show_ << "\t";

    return os;
}

std::istream& operator>>(std::istream& is, SparseAdamValue& value) {
    is >> value.dim_;

    for (int i = 0; i < value.dim_; i++) {
        is >> value.Weight()[i];
    }

    for (int i = 0; i < value.dim_; i++) {
        is >> value.M()[i];
    }

    for (int i = 0; i < value.dim_; i++) {
        is >> value.V()[i];
    }

    is >> value.version_;
    is >> value.show_;

    return is;
}

} // namespace tensornet {

