#!/usr/bin/python3.6
# coding=utf-8
import argparse
from pyspark.sql import SparkSession
from utils import output_line
from utils import load_sparse_table_to_df


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
    spark = SparkSession.builder.appName("[spark][merge sparse table]").master("yarn").enableHiveSupport().getOrCreate()

    sc = spark.sparkContext
    dims_df = load_sparse_table_to_df(sc, args.input, args.format)
    dims_df.select("sign", "weights", "handle").dropDuplicates(["sign", "handle"]).drop("handle").rdd.map(
        lambda x: output_line(x, args.bracket)
    ).repartition(args.number).saveAsTextFile(args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
