load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
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
    alwayslink = 0,
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
    name = "bn_table_ops_kernels",
    srcs = [
        "kernels/bn_table_ops_dummy.cc",
        "ops/bn_table_ops.cc",
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
    alwayslink = 0,
)

tf_gen_op_wrapper_py(
    name = "bn_table_ops",
    deps = [":bn_table_ops_kernels"],
    cc_linkopts = ['-lrt']
)

cc_library(
    name = "balance_dataset_ops_kernels",
    srcs = [
        "kernels/data/balance_dataset_ops_dummy.cc",
        "ops/balance_dataset_ops.cc",
    ],
    hdrs = [
        "kernels/data/balance_dataset_ops_dummy.h",
        "public/version.h",
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
        "//core/ps_interface:ps_raw_interface",
    ],
    deps = [
        "//core/ps_interface:server_cc_proto",
        ":_ps_optimizer",
        "@brpc//:brpc",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "_pywrap_tn.so",
    srcs = glob([
        "ps/*.cc",
        "ops/*.cc",
        "main/*.cc",
        "ps/*.h",
    ]) + [
        "kernels/sparse_table_ops.cc",
        "kernels/dense_table_ops.cc",
        "kernels/data/balance_dataset_ops.cc",
        "kernels/data/balance_dataset_ops.h",
		"kernels/bn_table_ops.cc",
        "public/version.h",
        "kernels/resource_var_wrapper.h",
        "//core/utility:semaphore",
        "//core/ps_interface:ps_raw_interface",
    ],
    deps = [
        "//core/ps_interface:server_cc_proto",
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
    ],
    linkopts = [
     "-Wl,-rpath,$$ORIGIN/../libs",
     "-Wl,-rpath,$$ORIGIN/../../../../../lib",
    ],
    linkshared = 1,
)
