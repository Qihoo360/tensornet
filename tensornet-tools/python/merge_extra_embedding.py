#coding=utf-8
import sys
import argparse
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import col, udf, lit
from pyspark.sql import functions as F
from pyspark.sql.types import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="sparse table input path")
    parser.add_argument("-o", "--output", type=str, help="merged file output path")
    parser.add_argument("-f", "--format", type=str, help="input file format, 'txt' or 'bin'")
    parser.add_argument("-e", "--extra", type=str, help="extra embedding file path")
    args = parser.parse_args()
    return args


def main(args):
    spark = SparkSession.builder \
        .appName("[spark][merge extra embedding]") \
        .master('yarn') \
        .enableHiveSupport() \
        .getOrCreate()

    sc = spark.sparkContext
    output_bc_value = sc.broadcast(args.output)
    format_bc_value = sc.broadcast(args.format)

    handle_names = fetch_hanlds(args.input)
    handle_names_bc_value = sc.broadcast(handle_names)

    source_rank_num = fetch_input_rank_num(args.input)
    number_bc_value = sc.broadcast(source_rank_num)

    get_sign_partition_key_udf = udf(get_sign_partition_key, IntegerType())

    dims_df = load_sparse_table_to_df(sc, args.input, args.format).withColumn('par_key', get_sign_partition_key_udf(col('sign'), lit(source_rank_num)))

    extra_data_df = sc.textFile(args.extra).map(lambda x: (x.split(',')[0], x.split(',')[1].split(':')[0], get_sign_partition_key(x.split(',')[1].split(':')[0], source_rank_num), x.split(',')[1].split(':')[1], get_sign_partition_key(x.split(',')[1].split(':')[1], source_rank_num))).toDF(['handle_name', 'old_sign', 'old_sign_par_key', 'new_sign', 'new_sign_par_key'])

    extra_data_all_df = extra_data_df.join(dims_df, (extra_data_df.handle_name == dims_df.handle) & (extra_data_df.old_sign == dims_df.sign) & (extra_data_df.old_sign_par_key == dims_df.par_key)).selectExpr("new_sign as sign", "dim", "weights", "g2sum", "show", "no_show_days", "handle", "new_sign_par_key as par_key")
   
    dims_df.unionAll(extra_data_all_df).rdd.map(lambda row: (row[7], row)).partitionBy(source_rank_num * BLOCK_NUM)\
      .foreachPartition(lambda p: resize_partition(p, output_bc_value, format_bc_value, number_bc_value, handle_names_bc_value))


if __name__ == '__main__':
    args = parse_args()
    main(args)
