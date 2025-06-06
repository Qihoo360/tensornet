# Copyright 2020-2025 Qihoo Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import IntegerType
from utils import load_sparse_table_to_df
from utils import BLOCK_NUM
from utils import resize_partition
from utils import sparse_df_schema
from utils import get_weight_for_extra_embedding
from utils import get_sign_partition_key
from utils import SparseTablePathInfo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="sparse table input path")
    parser.add_argument("-o", "--output", type=str, help="merged file output path")
    parser.add_argument("-f", "--format", type=str, help="input file format, 'txt' or 'bin'")
    parser.add_argument("-e", "--extra", type=str, help="extra embedding file path")
    args = parser.parse_args()
    return args


def main(args):
    spark = (
        SparkSession.builder.appName("[spark][merge extra embedding]").master("yarn").enableHiveSupport().getOrCreate()
    )

    sc = spark.sparkContext
    output_bc_value = sc.broadcast(args.output)
    format_bc_value = sc.broadcast(args.format)
    path_info = SparseTablePathInfo(args.input)
    source_rank_num = path_info.total_rank_num
    handle_names = path_info.handles
    sparse_table_parent = path_info.sparse_parent
    handle_names_bc_value = sc.broadcast(handle_names)
    number_bc_value = sc.broadcast(source_rank_num)
    get_sign_partition_key_udf = udf(get_sign_partition_key, IntegerType())

    dims_df = load_sparse_table_to_df(sc, args.input, args.format).withColumn(
        "par_key", get_sign_partition_key_udf(col("sign"), lit(source_rank_num))
    )

    extra_data_rdd = (
        sc.textFile(args.extra)
        .map(
            lambda x: (
                x.split(",")[0],
                x.split(",")[1].split(":")[0],
                get_sign_partition_key(x.split(",")[1].split(":")[0], source_rank_num),
                x.split(",")[1].split(":")[1],
                get_sign_partition_key(x.split(",")[1].split(":")[1], source_rank_num),
            )
        )
        .map(lambda x: ((x[0], x[2]), x))
    )

    distinct_key_list = extra_data_rdd.keys().distinct().collect()
    repartition_num = len(distinct_key_list)

    dims_df.unionAll(
        extra_data_rdd.map(lambda x: (distinct_key_list.index(x[0]), x[1]))
        .partitionBy(repartition_num)
        .mapPartitions(lambda p: get_weight_for_extra_embedding(p, source_rank_num, sparse_table_parent))
        .toDF(sparse_df_schema)
    ).rdd.map(lambda row: (row[7], row)).partitionBy(source_rank_num * BLOCK_NUM).foreachPartition(
        lambda p: resize_partition(p, output_bc_value, format_bc_value, number_bc_value, handle_names_bc_value)
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
