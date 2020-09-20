cc_test(
    name = "optimizer_test",
    srcs = [
        "optimizer_kernel_test.cc",
        "//core/utility:random",
    ],
    deps = [
        "//core:_ps_optimizer",
        "@brpc//:brpc",
        "@com_google_googletest//:gtest_main",
    ],
    copts = ["-g -ggdb"],
)
