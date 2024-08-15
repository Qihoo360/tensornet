workspace(name = "tensornet")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    sha256 = "69cd836f87b8c53506c4f706f655d423270f5a563b76dc1cfa60fbc3184185a3",
    strip_prefix = "tensorflow-2.2.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.2.0.tar.gz",
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "brpc",
    urls = [
    "https://github.com/apache/brpc/archive/0.9.7.tar.gz"
    ],
    strip_prefix = "brpc-0.9.7",
    patches = ["//thirdparty/patches:01-fix_dl_sym.patch"],
    patch_args = ["-p", "1"],
)

# depend by brpc
http_archive(
    name = "com_github_google_leveldb",
    build_file = "@brpc//:leveldb.BUILD",
    strip_prefix = "leveldb-a53934a3ae1244679f812d998a4f16f2c7f309a6",
    url = "https://github.com/google/leveldb/archive/a53934a3ae1244679f812d998a4f16f2c7f309a6.tar.gz"
)

http_archive(
        name = "com_github_nelhage_rules_boost",
        urls = [
        "https://github.com/nelhage/rules_boost/archive/fe9a0795e909f10f2bfb6bfa4a51e66641e36557.tar.gz",
        ],
        strip_prefix = "rules_boost-fe9a0795e909f10f2bfb6bfa4a51e66641e36557",
    )

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

# below is needed by generate op python file

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")
bazel_version_repository(name = "bazel_version")

