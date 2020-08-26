#!/bin/bash

readonly WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly THIS_FILE="${WORKSPACE_DIR}/$(basename "${BASH_SOURCE[0]}")"

pushd $WORKSPACE_DIR > /dev/null

set -o pipefail
set -o xtrace

saved_model_cli aot_compile_cpu --dir ${WORKSPACE_DIR}/graph_variable/ \
                                --tag_set serve \
                                --output_prefix ${WORKSPACE_DIR}/graph \
                                --cpp_class Graph

slot_num=`grep "static constexpr size_t kNumArgs" ${WORKSPACE_DIR}/graph.h | awk '{print $NF}' | awk -F';' '{print $1}'`

python3 gen_graph.py ${slot_num} > ${WORKSPACE_DIR}/graph.cc
if [ $? -ne 0 ]; then
    echo "gen_graph.py ${slot_num} wrong."
    exit 1
fi
