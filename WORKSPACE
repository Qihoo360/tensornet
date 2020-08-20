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

# copy from @org_tensorflow/WORKSPACE
# TensorFlow build depends on these dependencies.
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
    "https://github.com/apache/incubator-brpc/archive/0.9.7.tar.gz"
    ],
    sha256 = "722cd342baf3b05189ca78ecf6c56ea6ffec22e62fc2938335e4e5bab545a49c",
    strip_prefix = "incubator-brpc-0.9.7",
)

# depend by brpc
http_archive(
    name = "com_github_google_leveldb",
    build_file = "@brpc//:leveldb.BUILD",
    strip_prefix = "leveldb-a53934a3ae1244679f812d998a4f16f2c7f309a6",
    url = "https://github.com/google/leveldb/archive/a53934a3ae1244679f812d998a4f16f2c7f309a6.tar.gz"
)

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "fe9a0795e909f10f2bfb6bfa4a51e66641e36557",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1570056263 -0700",
)

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    commit = "bc2d0935b74917be0821bfd834472ed9cc4a3b5b",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

