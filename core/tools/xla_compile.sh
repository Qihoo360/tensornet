#!/bin/bash

readonly WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly THIS_FILE="${WORKSPACE_DIR}/$(basename "${BASH_SOURCE[0]}")"

pushd $WORKSPACE_DIR > /dev/null

set -o pipefail
set -o xtrace

saved_model_cli aot_compile_cpu --dir ${WORKSPACE_DIR}/graph_variable/ \
                                --variables_to_feed all \
                                --tag_set serve \
                                --output_prefix ${WORKSPACE_DIR}/graph \
                                --cpp_class Graph

mv ${WORKSPACE_DIR}/graph.o ${WORKSPACE_DIR}/graph_c.o
