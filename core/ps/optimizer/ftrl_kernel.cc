// Copyright 2020-2025 Qihoo Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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

DenseFtrlValue::DenseFtrlValue(const Ftrl* opt, int len) { return; }

void DenseFtrlValue::SetWeight(butil::IOBuf& w_buf) { return; }

void DenseFtrlValue::Apply(const Ftrl* opt, const Eigen::ArrayXf& g, const float lr) { return; }

std::ostream& operator<<(std::ostream& os, const DenseFtrlValue& value) { return os; }

std::istream& operator>>(std::istream& is, DenseFtrlValue& value) { return is; }

SparseFtrlValue::SparseFtrlValue(int dim, const Ftrl* opt) {
    float* w = Weight();
    float* z = Z(dim);
    float* n = N(dim);

    if (opt->sparse_zero_init_) {
        for (int i = 0; i < dim; ++i) {
            w[i] = 0.0;
            z[i] = 0;
            n[i] = 0;
        }
    } else {
        auto& reng = local_random_engine();
        auto distribution = std::normal_distribution<float>(0, 1 / sqrt(dim));

        for (int i = 0; i < dim; ++i) {
            w[i] = distribution(reng) * opt->initial_scale;
            z[i] = 0;
            n[i] = 0;
        }
    }
}

void SparseFtrlValue::Apply(const Ftrl* opt, SparseGradInfo& grad_info, int dim) {
    delta_show_ += grad_info.batch_show;

    float* w = Weight();
    float* z = Z(dim);
    float* n = N(dim);

    for (int i = 0; i < dim; ++i) {
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

void SparseFtrlValue::SerializeTxt_(std::ostream& os, int dim) {
    float* w = Weight();
    float* z = Z(dim);
    float* n = N(dim);

    for (int i = 0; i < dim; i++) {
        os << w[i] << "\t";
        os << z[i] << "\t";
        os << n[i] << "\t";
    }

    os << show_;
}

void SparseFtrlValue::DeSerializeTxt_(std::istream& is, int dim) {
    float* w = Weight();
    float* z = Z(dim);
    float* n = N(dim);

    for (int i = 0; i < dim; i++) {
        is >> w[i];
        is >> z[i];
        is >> n[i];
    }

    is >> show_;
}

void SparseFtrlValue::SerializeBin_(std::ostream& os, int dim) {
    os.write(reinterpret_cast<const char*>(Weight()), dim * sizeof(float));
    os.write(reinterpret_cast<const char*>(Z(dim)), dim * sizeof(float));
    os.write(reinterpret_cast<const char*>(N(dim)), dim * sizeof(float));
    os.write(reinterpret_cast<const char*>(&show_), sizeof(show_));
}

void SparseFtrlValue::DeSerializeBin_(std::istream& is, int dim) {
    is.read(reinterpret_cast<char*>(Weight()), dim * sizeof(float));
    is.read(reinterpret_cast<char*>(Z(dim)), dim * sizeof(float));
    is.read(reinterpret_cast<char*>(N(dim)), dim * sizeof(float));
    is.read(reinterpret_cast<char*>(&show_), sizeof(show_));
}

void SparseFtrlValue::ShowDecay(const Ftrl* opt, int /*delta_days*/) { show_ *= opt->show_decay_rate; }

bool SparseFtrlValue::DeleteByShow(const Ftrl* opt) { return show_ < opt->show_threshold; }

}  // namespace tensornet
