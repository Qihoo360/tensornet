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

#include "core/ps/optimizer/ada_grad_kernel.h"
#include "core/utility/allocator.h"

#include <butil/time.h>

using namespace tensornet;

TEST(allocator, perf) {
    float epsilon = 1e-8;
    float grad_decay_rate = 1.0;
    float mom_decay_rate = 1.0;
    float show_decay_rate = 0.98;

    // AdaGrad opt(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);
    AdaGrad opt;

    int dim = 8;
    Allocator<SparseAdaGradValue> alloc(SparseAdaGradValue::DynSizeof(dim));

    butil::Timer timer(butil::Timer::STARTED);

    for (int i = 0; i < 1000; i++) {
        auto value = alloc.allocate(dim, &opt);
    }

    timer.stop();

    LOG(INFO) << "allocate time:" << timer.u_elapsed();

    EXPECT_LT(timer.u_elapsed(), 10000);
}
