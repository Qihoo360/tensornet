cc_library(
    name = "tensorflow",
    srcs = [
        "lib/libtensorflow_framework.so.2"
    ],
    includes = ["include"],
    hdrs = glob(["include/**"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorflow_py",
    srcs = [
        "lib/lib_pywrap_tensorflow_internal.so"
    ],
    visibility = ["//visibility:public"],
)
