
filegroup(
    name = "ps_raw_interface",
    srcs = [
        "ps_raw_interface.h",
    ],
    visibility = ["//visibility:public"]
)

proto_library(
    name = "server_proto",
    srcs = ["ps_server.proto"],
)

cc_proto_library(
    name = "server_cc_proto",
    deps = [":server_proto"],
    visibility = ["//visibility:public"]
)
