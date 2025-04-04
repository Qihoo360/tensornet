#!/usr/bin/env bash

WORKSPACE_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly WORKSPACE_DIR

cd -- "$WORKSPACE_DIR" || exit $?

# set -o errexit
set -o pipefail
#exit when your script tries to use undeclared variables
set -o nounset

function do_package()
{
    echo "do package"

    (
        set -o xtrace

        rm -rf "${WORKSPACE_DIR}"/tensornet*

        cp -r "${WORKSPACE_DIR}/../tensornet" "${WORKSPACE_DIR}"

        cp -f "${WORKSPACE_DIR}"/../bazel-bin/core/_pywrap_tn.so "${WORKSPACE_DIR}"/tensornet/core/

        rm "${WORKSPACE_DIR}"/tensornet/core/.gitignore

        find "${WORKSPACE_DIR}"/tensornet -name "__pycache__" -exec rm -rf {} +

        tar -zcvf tensornet.tar.gz tensornet
    )

    return 0
}

do_package
