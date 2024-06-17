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
#include <cstdlib>

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
    float* w = Weight();
    auto spare_env = std::getenv("SPARSE_INIT_ZERO");
    if (spare_env != nullptr) {
        for (int i = 0; i < dim; ++i) {
            w[i] = 0.0;
        }
    } else {
        auto& reng = local_random_engine();
        auto distribution = std::normal_distribution<float>(0, 1 / sqrt(dim));

        for (int i = 0; i < dim; ++i) {
            w[i] = distribution(reng) * opt->initial_scale;
        }
    }

    g2sum_ = opt->initial_g2sum;
    old_compat_ = false;
    no_show_days_ = 0;
}

void SparseAdaGradValue::Apply(const AdaGrad* opt, SparseGradInfo& grad_info, int dim) {
    show_ += grad_info.batch_show;
    no_show_days_ = 0;

    float* w = Weight();

    double add_g2sum = 0;

    for (int i = 0; i < dim; ++i) {
        add_g2sum += grad_info.grad[i] * grad_info.grad[i];
    }

    g2sum_ += add_g2sum / dim;

    for (int i = 0; i < dim; ++i) {
        w[i] -= opt->learning_rate * grad_info.grad[i] / (opt->epsilon + sqrt(g2sum_));
    }
}

void SparseAdaGradValue::SerializeTxt_(std::ostream& os, int dim) {
    os << dim << "\t";
    for (int i = 0; i < dim; i++) {
        os << Weight()[i] << "\t";
    }

    os << g2sum_ << "\t";
    os << show_ << "\t";
    os << no_show_days_;
}

void SparseAdaGradValue::DeSerializeTxt_(std::istream& is, int dim) {
    is >> dim_;
    for (int i = 0; i < dim; i++) {
        is >> Weight()[i];
    }

    is >> g2sum_;
    is >> show_;
    if(!old_compat_) {
        is >> no_show_days_;
    }
}

void SparseAdaGradValue::SerializeBin_(std::ostream& os, int dim) {
    os.write(reinterpret_cast<const char*>(Weight()), dim * sizeof(float));
    os.write(reinterpret_cast<const char*>(&g2sum_), sizeof(g2sum_));
    os.write(reinterpret_cast<const char*>(&show_), sizeof(show_));
    os.write(reinterpret_cast<const char*>(&no_show_days_), sizeof(no_show_days_));
}

void SparseAdaGradValue::DeSerializeBin_(std::istream& is, int dim) {
    is.read(reinterpret_cast<char*>(Weight()), dim * sizeof(float));
    is.read(reinterpret_cast<char*>(&g2sum_), sizeof(g2sum_));
    is.read(reinterpret_cast<char*>(&show_), sizeof(show_));
    if(!old_compat_) {
        is.read(reinterpret_cast<char*>(&no_show_days_), sizeof(no_show_days_));
    }
}

void SparseAdaGradValue::ShowDecay(const AdaGrad* opt, int delta_days) {
    show_ *= opt->show_decay_rate;
    no_show_days_ += delta_days;
}

bool SparseAdaGradValue::DeleteByShow(const AdaGrad* opt) {
    return show_ < opt->show_threshold || no_show_days_ > opt->no_show_days;
}

} // namespace tensornet

