#coding=utf-8
import sys
import argparse
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="sparse table input path")
    parser.add_argument("-o", "--output", type=str, help="merged file output path")
    parser.add_argument("-f", "--format", type=str, help="input file format, 'txt' or 'bin'")
    parser.add_argument("-n", "--number", type=int, help="output file parallelism", default=30)
    args = parser.parse_args()
    return args


def main(args):
    spark = SparkSession.builder \
        .appName("[spark][resize sparse table]") \
        .master('yarn') \
        .enableHiveSupport() \
        .getOrCreate()

    sc = spark.sparkContext
    output_bc_value = sc.broadcast(args.output)
    format_bc_value = sc.broadcast(args.format)
    number_bc_value = sc.broadcast(args.number)

    handle_names = fetch_hanlds(args.input)
    handle_names_bc_value = sc.broadcast(handle_names)

    dims_df = load_sparse_table_to_df(sc, args.input, args.format)

    dims_df.rdd.map(lambda row: (get_sign_partition_key(row[0], args.number), row)).partitionBy(args.number * BLOCK_NUM)\
      .foreachPartition(lambda p: resize_partition(p, output_bc_value, format_bc_value, number_bc_value, handle_names_bc_value))


if __name__ == '__main__':
    args = parse_args()
    main(args)
