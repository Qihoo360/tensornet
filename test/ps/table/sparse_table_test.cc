#include <gtest/gtest.h>

#include "core/utility/random.h"
#include "core/ps/table/sparse_table.h"

#include <butil/logging.h>

using namespace tensornet;

class SparseTableTest : public testing::Test {
public:
    SparseTableTest() {
        float epsilon = 1e-8;
        float grad_decay_rate = 1.0;
        float mom_decay_rate = 1.0;
        float show_decay_rate = 0.98;

        opt = new AdaGrad(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);
    }

    ~SparseTableTest() {
        delete opt;
    }

    virtual void SetUp() {
        table = CreateSparseTable(opt, dim, shard_num, shard_id);
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
        auto sign_info = req.add_sign_infos();
        sign_info->set_var_index(i);
        sign_info->set_index(i);

        uint64_t sign = distr(reng);

        sign_info->set_sign(sign);
    }

    butil::Timer timer(butil::Timer::STARTED);

    SparsePullResponse resp;

    table->Pull(&req, &resp);

    timer.stop();

    LOG(INFO) << "Pull time:" << timer.u_elapsed();

    //EXPECT_LT(timer.u_elapsed(), 10000);
}

