#!/bin/bash
#
# This file is open source software, licensed to you under the terms
# of the Apache License, Version 2.0 (the "License").  See the NOTICE file
# distributed with this work for additional information regarding copyright
# ownership.  You may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# os-release may be missing in container environment by default.
if [ -f "/etc/os-release" ]; then
    . /etc/os-release
elif [ -f "/etc/arch-release" ]; then
    export ID=arch
else
    echo "/etc/os-release missing."
    exit 1
fi

debian_packages=(
    ninja-build
    ragel
    libhwloc-dev
    libnuma-dev
    libpciaccess-dev
    libcrypto++-dev
    libboost-all-dev
    libxml2-dev
    xfslibs-dev
    libgnutls28-dev
    liblz4-dev
    libsctp-dev
    gcc
    make
    libprotobuf-dev
    protobuf-compiler
    python3
    python3-pip
    systemtap-sdt-dev
    libtool
    cmake
    libyaml-cpp-dev
    libc-ares-dev
    stow
    g++
    libfmt-dev
    diffutils
    valgrind
    doxygen
    libssl-dev
    git
    curl
    wget
    pkg-config
)

# seastar doesn't directly depend on these packages. They are
# needed because we want to link seastar statically and pkg-config
# has no way of saying "static seastar, but dynamic transitive
# dependencies". They provide the various .so -> .so.ver symbolic
# links.
transitive=(libtool-ltdl-devel trousers-devel libidn2-devel libunistring-devel)

redhat_packages=(
    hwloc-devel
    numactl-devel
    libpciaccess-devel
    cryptopp-devel
    libxml2-devel
    xfsprogs-devel
    gnutls-devel
    lksctp-tools-devel
    lz4-devel
    gcc
    make
    protobuf-devel
    protobuf-compiler
    python3
    python3-pip
    systemtap-sdt-devel
    libtool
    cmake
    yaml-cpp-devel
    c-ares-devel
    stow
    diffutils
    doxygen
    libssl-dev
    git
    curl
    wget
    "${transitive[@]}"
)

fedora_packages=(
    "${redhat_packages[@]}"
    gcc-c++
    ninja-build
    ragel
    boost-devel
    fmt-devel
    libubsan
    libasan
    libatomic
    valgrind-devel
)

centos7_packages=(
    "${redhat_packages[@]}"
    ninja-build
    ragel
    cmake3
    rh-mongodb36-boost-devel
    devtoolset-9-gcc-c++
    devtoolset-9-libubsan
    devtoolset-9-libasan
    devtoolset-9-libatomic
)

centos8_packages=(
    "${redhat_packages[@]}"
    ninja-build
    ragel
    gcc-toolset-9-gcc
    gcc-toolset-9-gcc-c++
    gcc-toolset-9-libubsan-devel
    gcc-toolset-9-libasan-devel
    gcc-toolset-9-libatomic-devel
)

case "$ID" in
    ubuntu|debian)
        apt-get install -y "${debian_packages[@]}"
    ;;
    fedora)
        dnf install -y "${fedora_packages[@]}"
    ;;
    centos)
        if [ "$VERSION_ID" = "7" ]; then
            yum install -y epel-release centos-release-scl scl-utils
            yum install -y "${centos7_packages[@]}"
        elif [ "$VERSION_ID" = "8" ]; then
            dnf install -y epel-release
            dnf install -y "${centos8_packages[@]}"
        fi
    ;;
    *)
        echo "Your system ($ID) is not supported by this script. Please install dependencies manually."
        exit 1
    ;;
esac
