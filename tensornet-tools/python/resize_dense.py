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
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="sparse table input path")
    parser.add_argument("-o", "--output", type=str, help="merged file output path")
    parser.add_argument("-n", "--number", type=int, help="output file parallelism", default=30)
    args = parser.parse_args()
    return args


def main(args):
    spark = SparkSession.builder \
        .appName("[spark][resize dense table]") \
        .master('yarn') \
        .enableHiveSupport() \
        .getOrCreate()

    sc = spark.sparkContext
    output_bc_value = sc.broadcast(args.output)
    dense_file_rdd = sc.wholeTextFiles(args.input).map(lambda x: (x[0].split("/")[-1], x[0].split("/")[-2], x[1])).flatMap(mapIndexToDenseRecord)

    whole_data = dense_file_rdd.collect()
    res = process_whole_text(whole_data, args.number)

    res_rdd = sc.parallelize(res, args.number)
    res_rdd.foreachPartition(lambda p:write_dense_partition(p, output_bc_value))


if __name__ == '__main__':
    args = parse_args()
    main(args)
