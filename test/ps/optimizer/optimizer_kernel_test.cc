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

#include <gtest/gtest.h>

#include "core/ps/optimizer/optimizer_kernel.h"
#include "core/utility/random.h"

#include <butil/time.h>

using namespace tensornet;

TEST(optimizer, GetWeightPerf) {
    float epsilon = 1e-8;
    float grad_decay_rate = 1.0;
    float mom_decay_rate = 1.0;
    float show_decay_rate = 0.98;

    // AdaGrad opt(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);
    AdaGrad opt;

    int dim = 8;
    auto op_kernel = opt.CreateSparseOptKernel(dim);

    auto& reng = local_random_engine();
    std::uniform_int_distribution<uint64_t> distr;

    butil::Timer timer(butil::Timer::STARTED);

    for (int i = 0; i < 1000; i++) {
        uint64_t sign = distr(reng);
        float* w = op_kernel->GetWeight(sign);
    }

    timer.stop();

    LOG(INFO) << "GetWeight time:" << timer.u_elapsed();

    // EXPECT_LT(timer.u_elapsed(), 10000);
}
