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

#include "core/ps/optimizer/ftrl_kernel.h"

#include <butil/logging.h>
#include <butil/rand_util.h>

#include "core/utility/random.h"

namespace tensornet {

DenseFtrlValue::DenseFtrlValue(const Ftrl* opt, int len) {
    return;
}

void DenseFtrlValue::SetWeight(butil::IOBuf& w_buf) {
    return;
}

void DenseFtrlValue::Apply(const Ftrl* opt, const Eigen::ArrayXf& g) {
    return;
}

std::ostream& operator<<(std::ostream& os, const DenseFtrlValue& value) {
    return os;
}

std::istream& operator>>(std::istream& is, DenseFtrlValue& value) {
    return is;
}

SparseFtrlValue::SparseFtrlValue(int dim, const Ftrl* opt) {
    dim_ = dim;

    auto& reng = local_random_engine();
    auto distribution = std::normal_distribution<float>(0, 1 / sqrt(Dim()));

    for (int i = 0; i < Dim(); ++i) {
        Weight()[i] = distribution(reng) * opt->initial_range;
        Z()[i] = 0;
        N()[i] = 0;
    }
}

void SparseFtrlValue::Apply(const Ftrl* opt, SparseGradInfo& grad_info) {
    float* w = Weight();
    float* z = Z();
    float* n = N();

    for (int i = 0; i < dim_; ++i) {
        float g2 = grad_info.grad[i] * grad_info.grad[i];

        z[i] += grad_info.grad[i] - opt->learning_rate * (sqrt(n[i] + g2) - sqrt(n[i])) * w[i];
        n[i] += g2;
        if (abs(z[i]) <= opt->lambda1) {
            w[i] = 0;
        } else {
            w[i] = -1 / ((opt->beta + sqrt(n[i])) * opt->learning_rate + opt->lambda2);
            if (z[i] > 0) {
                w[i] *= z[i] - opt->lambda1;
            } else {
                w[i] *= z[i] + opt->lambda1;
            }
        }
    }
}

std::ostream& operator<<(std::ostream& os, const SparseFtrlValue& value) {
    os << value.dim_ << "\t";

    for (int i = 0; i < value.dim_; i++) {
        os << value.Weight()[i] << "\t";
        os << value.Z()[i] << "\t";
        os << value.N()[i] << "\t";
    }

    os << value.show_;

    return os;
}

std::istream& operator>>(std::istream& is, SparseFtrlValue& value) {
    int dim;
    is >> dim;

    CHECK_EQ(dim, value.dim_);

    for (int i = 0; i < value.dim_; i++) {
        is >> value.Weight()[i];
        is >> value.Z()[i];
        is >> value.N()[i];
    }

    is >> value.show_;

    return is;
}

void SparseFtrlValue::ShowDecay(const Ftrl* opt) {
    show_ *= opt->show_decay_rate;
}

bool SparseFtrlValue::DeleteByShow(const Ftrl* opt) {
    return show_ < opt->show_threshold;
}

} // namespace tensornet

