load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "pybind_extension",
    "tf_gen_op_wrapper_py",
)

proto_library(
    name = "server_proto",
    srcs = ["protobuf/ps_server.proto"],
)

cc_proto_library(
    name = "server_cc_proto",
    deps = [":server_proto"],
)

cc_library(
    name = "sparse_table_ops_kernels",
    srcs = [
        "kernels/sparse_table_ops_dummy.cc",
        "ops/sparse_table_ops.cc",
    ],
    hdrs = [
        "//core/utility:semaphore",
    ],
    linkstatic = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "sparse_table_ops",
    deps = [":sparse_table_ops_kernels"],
    cc_linkopts = ['-lrt']
)

cc_library(
    name = "dense_table_ops_kernels",
    srcs = [
        "kernels/dense_table_ops_dummy.cc",
        "ops/dense_table_ops.cc",
    ],
    hdrs = [
        "//core/utility:semaphore",
    ],
    linkstatic = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "dense_table_ops",
    deps = [":dense_table_ops_kernels"],
    cc_linkopts = ['-lrt'],
)

cc_library(
    name = "balance_dataset_ops_kernels",
    srcs = [
        "kernels/data/balance_dataset_ops_dummy.cc",
        "ops/balance_dataset_ops.cc",
    ],
    hdrs = [
        "kernels/data/balance_dataset_ops_dummy.h",
    ],
    linkstatic = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/kernels/data:name_utils",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "balance_dataset_ops",
    deps = [":balance_dataset_ops_kernels"],
    cc_linkopts = ['-lrt'],
)

cc_library(
    name = "_ps_optimizer",
    srcs = glob([
        "ps/optimizer/*.cc",
    ]),
    hdrs = glob([
        "ps/optimizer/*.h",
    ]) + [
        "//core/utility:random",
        "//core/utility:allocator",
    ],
    deps = [
        "//core/utility:file_io",
        "@brpc//:brpc",
        "@boost//:iostreams",
        "@org_tensorflow//third_party/eigen3:eigen3",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "_ps_table",
    srcs = glob([
        "ps/table/*.cc",
    ]),
    hdrs = glob([
        "ps/table/*.h",
    ]) + [
    ],
    deps = [
        ":server_cc_proto",
        ":_ps_optimizer",
        "@brpc//:brpc",
    ],
    visibility = ["//visibility:public"],
)

pybind_extension(
    name = "_pywrap_tn",
    srcs = glob([
        "ps/*.cc",
        "ops/*.cc",
        "main/*.cc",
    ]) + [
        "kernels/sparse_table_ops.cc",
        "kernels/dense_table_ops.cc",
        "kernels/data/balance_dataset_ops.cc",
    ],
    hdrs = glob([
        "ps/*.h",
    ]) + [
        "kernels/data/balance_dataset_ops.h",
        "kernels/resource_var_wrapper.h",
        "//core/utility:semaphore",
    ],
    module_name = "_pywrap_tn",
    deps = [
        ":server_cc_proto",
        ":_ps_table",
        "//core/utility:mpi_manager",
        "//thirdparty/tensorflow:tensorflow",
        "//thirdparty/tensorflow:tensorflow_py",
        "//thirdparty/openmpi:openmpi",
        "@brpc//:brpc",
        "@pybind11",
        "@boost//:variant",
        "@boost//:random",
        "@boost//:iostreams",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/kernels/data:name_utils",
        "@org_tensorflow//tensorflow/python:pybind11_status",
        "@org_tensorflow//tensorflow/python:pybind11_lib",
    ]
)

filegroup(
    name = "graph",
    srcs = [
        "tools/graph.cc",
        "tools/graph_c.o",
        "tools/graph.h",
        "tools/object_pool.h",
        "tools/dual_dict.h",
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
        "@boost//:algorithm",
    ],
    linkshared = 1,
    linkopts = ["-lpthread"],
    copts = ["-fPIC"],
)
