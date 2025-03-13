#include <gtest/gtest.h>

#include "core/utility/random.h"
#include "core/ps/table/sparse_table.h"

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

        //opt = new AdaGrad(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);
        opt = new AdaGrad;
    }

    ~SparseTableTest() {
        delete opt;
    }

    virtual void SetUp() {
        table = CreateSparseTable(opt, "table-name", dim, shard_num, shard_id);
    }

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

    //EXPECT_LT(timer.u_elapsed(), 10000);
}

