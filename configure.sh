#!/bin/sh

readonly WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly THIS_FILE="${WORKSPACE_DIR}/$(basename "${BASH_SOURCE[0]}")"

pushd $WORKSPACE_DIR > /dev/null

readonly PYTHON_PATH="$(which python)"

function check_tf_version()
{
    echo "checking tensorflow version installed..."

    local tf_version=$(python -c "import tensorflow as tf; print(tf.version.VERSION)")

    if [[ "x${tf_version}" != "x2.2.0" ]]; then
        echo "tensorflow version is ${tf_version}, please use 2.2.0 instead"
        exit 1
    fi

    echo "tensorflow version installed is ${tf_version}"
}

function link_mpi_thirdparty()
{
    read -p "please give us your openmpi install path:" mpi_path

    echo "using openmpi include path:$mpi_path/include"
    echo "using openmpi lib path:$mpi_path/lib"

    rm -rf thirdparty/openmpi/include
    ln -s ${mpi_path}/include thirdparty/openmpi/

    rm -rf thirdparty/openmpi/lib
    ln -s ${mpi_path}/lib thirdparty/openmpi/
}

function link_tf_thirdparty()
{
    local tf_include_path=$(python -c "import tensorflow as tf;print(tf.sysconfig.get_include())")
    local tf_lib_path=$(python -c "import tensorflow as tf;print(tf.sysconfig.get_lib())")

    echo "using tensorflow lib path:${tf_lib_path}"

    rm thirdparty/tensorflow/lib/*
    ln -s ${tf_lib_path}/lib* thirdparty/tensorflow/lib/
    ln -sf ${tf_lib_path}/python/_pywrap_tensorflow_internal.so thirdparty/tensorflow/lib/lib_pywrap_tensorflow_internal.so
}

function main()
{
    echo "using python:${PYTHON_PATH}"

    check_tf_version
    link_mpi_thirdparty
    link_tf_thirdparty

    echo "configure done"
}

main $@

popd > /dev/null

exit 0

