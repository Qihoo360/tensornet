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

    AdaGrad opt(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);

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

    //EXPECT_LT(timer.u_elapsed(), 10000);
}

