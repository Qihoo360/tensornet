#include <gtest/gtest.h>

#include "core/utility/allocator.h"
#include "core/ps/optimizer/ada_grad_kernel.h"

#include <butil/time.h>

using namespace tensornet;

TEST(allocator, perf) {
    float epsilon = 1e-8;
    float grad_decay_rate = 1.0;
    float mom_decay_rate = 1.0;
    float show_decay_rate = 0.98;

    //AdaGrad opt(0.01, 0.1, 0.1, epsilon, grad_decay_rate, mom_decay_rate, show_decay_rate);
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
