filegroup(
    name = "graph",
    srcs = [
        "graph.cc",
        "graph.o",
        "graph.h",
    ],
)

cc_binary(
    name = "libmodel.so",
    srcs = [":graph"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@org_tensorflow//tensorflow/compiler/xla/service/cpu:runtime_conv2d",
        "@org_tensorflow//tensorflow/compiler/xla/service/cpu:runtime_key_value_sort",
        "@org_tensorflow//tensorflow/compiler/xla/service/cpu:runtime_matmul",
        "@org_tensorflow//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d",
        "@org_tensorflow//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "@org_tensorflow//third_party/eigen3:eigen3",
    ],
    linkshared = 1,
    linkopts = ["-lpthread"],
    copts = ["-fPIC"],
)

cc_binary(
    name = "tf_serving",
    srcs = [
        "main.cc",
        "//core:random",
    ],
    deps = [
        "@boost//:algorithm",
    ],
    linkopts = ["-ldl", "-lrt"],
)
