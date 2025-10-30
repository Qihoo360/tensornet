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

#include "core/ps/table/sparse_table.h"
#include "core/utility/random.h"

#include <brpc/controller.h>
#include <butil/logging.h>

using namespace tensornet;

class SparseTableTest : public testing::Test {
public:
    SparseTableTest() {
        float epsilon = 1e-8;
        float grad_decay_rate = 1.0;
        float mom_decay_rate = 1.0;
        float show_decay_rate = 0.98;

        // opt = new AdaGrad(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);
        opt = new AdaGrad;
    }

    ~SparseTableTest() { delete opt; }

    virtual void SetUp() { table = CreateSparseTable(opt, "table-name", dim, shard_num, shard_id, ""); }

    virtual void TearDown() {
        // nothing to be done, table will be delete by SparseTableRegistry deconstructor
    }

public:
    OptimizerBase* opt;
    SparseTable* table = nullptr;

    int dim = 8;
    int shard_num = 50;
    int shard_id = 0;
};

TEST_F(SparseTableTest, pull) {
    SparsePullRequest req;

    req.set_table_handle(table->GetHandle());
    req.set_dim(dim);

    auto& reng = local_random_engine();
    std::uniform_int_distribution<uint64_t> distr;

    for (int i = 0; i < 1000; i++) {
        req.add_signs(distr(reng));
    }

    butil::Timer timer(butil::Timer::STARTED);

    SparsePullResponse resp;
    brpc::Controller cntl;
    butil::IOBuf& output = cntl.response_attachment();
    table->Pull(&req, output, &resp);

    timer.stop();

    LOG(INFO) << "Pull time:" << timer.u_elapsed();

    // EXPECT_LT(timer.u_elapsed(), 10000);
}
