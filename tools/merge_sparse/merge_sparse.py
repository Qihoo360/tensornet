#!/usr/bin/python3.6
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
    parser.add_argument("-b", "--bracket", help="if dims need bracket", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(args):
    spark = SparkSession.builder \
        .appName("[spark][merge sparse table]") \
        .master('yarn') \
        .enableHiveSupport() \
        .getOrCreate()

    sc = spark.sparkContext

    if args.format == 'txt':
        get_handle_name_udf = udf(get_handle_name, StringType())
        dims_df = sc.textFile(args.input)\
                     .map(lambda x: process_txt_line(x))\
                     .toDF(["key", "dims"])\
                     .withColumn("input_file_name",F.input_file_name())\
                     .withColumn("handle", get_handle_name_udf(col("input_file_name")))\
                     .drop("input_file_name")\
                     .filter(col("key") != "").dropDuplicates(['key','handle'])
    elif args.format == 'bin':
        dims_df = sc.binaryFiles(args.input)\
                     .mapPartitions(process_binary_partition)\
                     .toDF(['handle', 'key', 'dims'])

    dims_df.dropDuplicates(['key','handle']).drop('handle').rdd.map(lambda x: output_line(x, args.bracket)).repartition(args.number).saveAsTextFile(args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
