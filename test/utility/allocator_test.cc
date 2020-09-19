#include <gtest/gtest.h>

#include "core/utility/allocator.h"

#include <butil/time.h>

struct Value {
    int a;
    float b;
    char data[0];
};

using namespace tensornet;

TEST(allocator, perf) {
    Allocator<Value> alloc(sizeof(Value));

    butil::Timer timer(butil::Timer::STARTED);

    for (int i = 0; i < 100000; i++) {
        Value* value = alloc.allocate();
    }

    timer.stop();

    LOG(INFO) << "allocate time:" << timer.u_elapsed();

    EXPECT_LT(timer.u_elapsed(), 10000);
}

