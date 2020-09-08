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
        "utility/semaphore.h",
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
        "utility/semaphore.h",
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

pybind_extension(
    name = "_pywrap_tn",
    srcs = glob([
        "ps/*.cc",
        "ps/optimizer/*.cc",
        "ps/table/*.cc",
        "ops/*.cc",
        "kernels/sparse_table_ops.cc",
        "kernels/dense_table_ops.cc",
        "kernels/data/balance_dataset_ops.cc",
        "main/*.cc",
        "utility/*.cc",
    ]),
    hdrs = glob([
        "ps/*.h",
        "ps/table/*.h",
        "ps/optimizer/*.h",
        "ps/hash/*.h",
        "utility/*.h",
        "kernels/data/balance_dataset_ops.h",
        "kernels/resource_var_wrapper.h",
    ]),
    module_name = "_pywrap_tn",
    deps = [
        ":server_cc_proto",
        "//thirdparty/tensorflow:tensorflow",
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
        "@org_tensorflow//third_party/eigen3:eigen3",
    ]
)
