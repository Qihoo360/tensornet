
filegroup(
    name = "random",
    srcs = [
        "random.h",
    ],
    visibility = ["//visibility:public"]
)

filegroup(
    name = "allocator",
    srcs = [
        "allocator.h",
    ],
    visibility = ["//visibility:public"]
)

filegroup(
    name = "semaphore",
    srcs = [
        "semaphore.h",
    ],
    visibility = ["//visibility:public"]
)

filegroup(
    name = "fix_redef",
    srcs = [
        "fix_redef.h",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "net_util",
    srcs = [
        "net_util.h",
        "net_util.cc",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "mpi_manager",
    srcs = [
        "mpi_manager.h",
        "mpi_manager.cc",
    ],
    deps = [
        ":net_util",
        "//thirdparty/openmpi:openmpi",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "file_io",
    srcs = [
        "file_io.h",
        "file_io.cc",
    ],
    deps = [
        "@boost//:iostreams",
        "//thirdparty/tensorflow:tensorflow",
    ],
    visibility = ["//visibility:public"]
)
