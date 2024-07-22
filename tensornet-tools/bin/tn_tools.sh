#!/usr/bin/env bash

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR=${WORKSPACE_DIR}/../python
TMP_PACKAGE_DIR=$(mktemp -d)
TN_TOOL_ENV_NAME=tn_tool
TN_TOOL_TGZ=${TMP_PACKAGE_DIR}/${TN_TOOL_ENV_NAME}.tar.gz

: ${SPARK_HOME:=/opt/spark3}

_die() {
  local err=$? err_fmt=
  (( err )) && err_fmt=" (err=$err)" || err=1
  printf >&2 "[ERROR]$err_fmt %s\n" "$*"
  exit $err
}

_check_spark_env(){

if [[ ! -d ${SPARK_HOME} ]] || [[ ! -e ${SPARK_HOME}/bin/spark-submit ]];then
    _die "no valid spark path, should export valid SPARK_HOME"
fi

spark_major_version=$($SPARK_HOME/bin/spark-submit --version 2>&1 | grep version | awk -F"version" '{print $2}' | head -1 | sed 's/ //g' | awk -F. '{print $1}')

if [[ -z ${spark_major_version} ]] || [[ $spark_major_version -lt 3 ]];then
    _die "invalid spark version. should be >= 3"
fi

}

_prepare_mamba_env(){
   if ! type micromamba >/dev/null 2>&1;then
     HTTPS_PROXY=${PROXY_URL:=${HTTPS_PROXY}} "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   fi
   _mamba_source
   [[ -z ${NEXUS3_HEADER} ]] || {
       ${MAMBA_EXE} config set --file "${MAMBA_ROOT_PREFIX}/.mambarc" channel_alias "${NEXUS3_HEADER}/conda"
   }
   micromamba create -y -f "${WORKSPACE_DIR}/config/tn_tool_env.yaml"
   micromamba activate ${TN_TOOL_ENV_NAME}
   TN_TOOL_ENV_DIR=$(micromamba env list | grep "${TN_TOOL_ENV_NAME}" | awk '{print $NF}')
   conda-pack --prefix ${TN_TOOL_ENV_DIR} -o ${TN_TOOL_TGZ}
}

start_merge_sparse(){

${SPARK_HOME}/bin/spark-submit --executor-memory 8g --driver-memory 10g --py-files ${PYTHON_DIR}/utils.py ${PYTHON_DIR}/merge_sparse.py "$@"

}

start_resize_sparse(){
_prepare_mamba_env

${SPARK_HOME}/bin/spark-submit --conf spark.executor.memory=10g --conf spark.archives=file://${TN_TOOL_TGZ}#envs --conf spark.pyspark.driver.python=${TN_TOOL_ENV_DIR}/bin/python --conf spark.pyspark.python=envs/bin/python  --py-files ${PYTHON_DIR}/utils.py ${PYTHON_DIR}/resize_sparse.py "$@"
}

start_resize_sparse(){
_prepare_mamba_env

${SPARK_HOME}/bin/spark-submit --conf spark.executor.memory=10g --conf spark.archives=file://${TN_TOOL_TGZ}#envs --conf spark.pyspark.driver.python=${TN_TOOL_ENV_DIR}/bin/python --conf spark.pyspark.python=envs/bin/python  --py-files ${PYTHON_DIR}/utils.py ${PYTHON_DIR}/resize_dense.py "$@"
}

_check_spark_env

case "${1-}" in
(merge-sparse)
  shift 1
  start_merge_sparse "$@"
  ;;
(resize-sparse)
  shift 1
  start_resize_sparse "$@"
  ;;
(resize-dense)
  shift 1
  start_resize_dense "$@"
  ;;
(''|help)
  cmd=$(basename -- "$0")
  cat <<-END
        Usage:
          $cmd [help] - Print this help.

          $cmd merge-sparse [-i/--input input_path] [-o/--output output_path] [-f/--format file_format] [-n/--number number] [-b/--bracket] - merge all tensornet generated sparse file into one hdfs directory.

          $cmd resize-sparse [-i/--input input_path] [-o/--output output_path] [-f/--format file_format] [-n/--number number] - change current sparse parallelism to another size.

          $cmd resize-dense [-i/--input input_path] [-o/--output output_path] [-n/--number number] - change current dense parallelism to another size.
END
  ;;
(*) die Unknown command "$1" ;;
esac

