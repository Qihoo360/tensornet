#!/bin/sh

readonly WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly THIS_FILE="${WORKSPACE_DIR}/$(basename "${BASH_SOURCE[0]}")"

pushd $WORKSPACE_DIR > /dev/null

readonly PYTHON_PATH="$(which python)"

# global parameter
OPENMPI_PATH=""

function _simple_help()
{
    echo -e "
    ${THIS_FILE} arguments

        arguments:
        --openmpi_path the path of your openmpi installed
        --help help info
        "

    return 0
}

function simple_eval_param()
{
    while [[ $# -gt 0 ]]; do
        key="$1"

        case $key in
            --openmpi_path)
                OPENMPI_PATH="$2" && shift
                ;;
            * |--help | -h)
                _simple_help
                exit 1
                ;;
        esac

        shift
    done

    echo "${OPENMPI_PATH}"
    if [ "x" == "x${OPENMPI_PATH}" ]; then
        echo "please specify where openmpi installed"
        _simple_help
        exit 1
    fi

    return 0
}

function check_tf_version()
{
    echo "checking tensorflow version installed..."

    local tf_version=$(python -c "import tensorflow as tf; print(tf.version.VERSION)")
    local tf_major_version=`echo ${tf_version} | awk -F'.' 'BEGIN{OFS="."}{print $1 OFS $2}'`

    if [[ "x${tf_major_version}" != "x2.2" ]] && [[ "x${tf_major_version}" != "x2.3" ]]; then
        echo "tensorflow version is ${tf_version}, please use 2.2.0 ~ 2.3.0 instead"
        exit 1
    fi

    echo "tensorflow version installed is ${tf_version}"
}

function link_mpi_thirdparty()
{
    echo "using openmpi include path:$OPENMPI_PATH/include"
    echo "using openmpi lib path:$OPENMPI_PATH/lib"

    rm -rf thirdparty/openmpi/include
    ln -s ${OPENMPI_PATH}/include thirdparty/openmpi/

    rm -rf thirdparty/openmpi/lib
    ln -s ${OPENMPI_PATH}/lib thirdparty/openmpi/
}

function link_tf_thirdparty()
{
    local tf_include_path=$(python -c "import tensorflow as tf;print(tf.sysconfig.get_include())")
    local tf_lib_path=$(python -c "import tensorflow as tf;print(tf.sysconfig.get_lib())")

    echo "using tensorflow lib path:${tf_lib_path}"

    rm thirdparty/tensorflow/lib/*
    mkdir -p thirdparty/tensorflow/lib/
    ln -s ${tf_lib_path}/lib* thirdparty/tensorflow/lib/
    ln -sf ${tf_lib_path}/python/_pywrap_tensorflow_internal.so thirdparty/tensorflow/lib/lib_pywrap_tensorflow_internal.so
    ln -sf ${tf_include_path} thirdparty/tensorflow/
}

function main()
{
    echo "using python:${PYTHON_PATH}"

    simple_eval_param $@

    check_tf_version
    link_mpi_thirdparty
    link_tf_thirdparty

    echo "configure done"
}

main $@

popd > /dev/null

exit 0
