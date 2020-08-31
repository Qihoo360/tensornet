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

#include "core/ps/optimizer/ada_grad_kernel.h"

#include <butil/logging.h>
#include <butil/rand_util.h>

#include "core/utility/random.h"

namespace tensornet {

DenseAdaGradValue::DenseAdaGradValue(const AdaGrad* opt, int len) {
    // NOTE, w must initialized as random. You can also call SetWeight initialize.
    // eigen init uniformly spread through [-1:1] default.
    w_.setRandom(len);
    w_ *= opt->initial_scale;

    d2sum_.setZero(len);
    g2sum_.setConstant(len, opt->initial_g2sum);
    m_.setZero(len);
}

void DenseAdaGradValue::SetWeight(butil::IOBuf& w_buf) {
    CHECK_EQ(w_.size() * sizeof(float), w_buf.size());

    w_buf.copy_to(w_.data(), w_.size() * sizeof(float));
}

void DenseAdaGradValue::Apply(const AdaGrad* opt, const Eigen::ArrayXf& g) {
    d2sum_ = opt->grad_decay_rate * d2sum_ + 1;
    g2sum_ = opt->grad_decay_rate * g2sum_ + g.square();

    m_ += (g - m_) * (1.0 - opt->mom_decay_rate);
    w_ -= opt->learning_rate * m_ / (g2sum_.sqrt() / d2sum_.sqrt() + opt->epsilon);
}

std::ostream& operator<<(std::ostream& os, const DenseAdaGradValue& value) {
    int array_size = value.w_.size();

    os << "arrary_size:" << array_size << std::endl;

    for (int i = 0; i < array_size; i++) {
        os << value.w_[i] << "\t"
           << value.d2sum_[i] << "\t"
           << value.g2sum_[i] << "\t"
           << value.m_[i] << std::endl;
    }

    return os;
}

std::istream& operator>>(std::istream& is, DenseAdaGradValue& value) {
    int array_size = 0;
    is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> array_size;

    CHECK_EQ(array_size, value.w_.size());

    for (int i = 0; i < array_size; i++) {
        is >> value.w_[i];
        is >> value.d2sum_[i];
        is >> value.g2sum_[i];
        is >> value.m_[i];
    }

    return is;
}

SparseAdaGradValue::SparseAdaGradValue(int dim, const AdaGrad* opt) {
    dim_ = dim;

    if (!IsMiniDim_()) {
        w_.p = new float[dim];
    }

    auto& reng = local_random_engine();
    auto distribution = std::normal_distribution<float>(0, 1 / sqrt(Dim()));

    for (int i = 0; i < Dim(); ++i) {
        if (IsMiniDim_()) {
            w_.v[i] = distribution(reng) * opt->initial_scale;
        } else {
            w_.p[i] = distribution(reng) * opt->initial_scale;
        }
    }

    g2sum_ = opt->initial_g2sum;
    version_ = 0;
}

void SparseAdaGradValue::Apply(const AdaGrad* opt, SparseGradInfo& grad_info) {
    CHECK_EQ(Dim(), grad_info.dim);

    version_++;
    show_ += grad_info.show;

    float* w = Weight();

    double add_g2sum = 0;
    double g_scale = grad_info.show;
    if (grad_info.version < Version()) {
        g_scale *= sqrt(Version() - grad_info.version);
    }

    for (int i = 0; i < dim_; ++i) {
        double scaled_grad = grad_info.grad[i] / g_scale;
        add_g2sum += scaled_grad * scaled_grad;
    }

    g2sum_ += add_g2sum / dim_;

    for (int i = 0; i < dim_; ++i) {
        double scaled_grad = grad_info.grad[i] / g_scale;
        w[i] -= opt->learning_rate * scaled_grad / (opt->epsilon + sqrt(g2sum_));
    }
}

std::ostream& operator<<(std::ostream& os, const SparseAdaGradValue& value) {
    os << value.dim_ << "\t";

    for (int i = 0; i < value.dim_; i++) {
        os << value.Weight()[i] << "\t";
    }

    os << value.g2sum_ << "\t";
    os << value.version_ << "\t";
    os << value.show_;

    return os;
}

std::istream& operator>>(std::istream& is, SparseAdaGradValue& value) {
    int dim;
    is >> dim;

    CHECK_EQ(dim, value.dim_);

    for (int i = 0; i < value.dim_; i++) {
        is >> value.Weight()[i];
    }

    is >> value.g2sum_;
    is >> value.version_;
    is >> value.show_;

    return is;
}

void SparseAdaGradValue::ShowDecay(const AdaGrad* opt) {
    show_ *= opt->show_decay_rate;
}

} // namespace tensornet

