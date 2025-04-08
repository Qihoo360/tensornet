# coding=utf-8
import argparse
from pyspark.sql import SparkSession
from utils import write_dense_partition
from utils import process_whole_text
from utils import mapIndexToDenseRecord


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="sparse table input path")
    parser.add_argument("-o", "--output", type=str, help="merged file output path")
    parser.add_argument("-n", "--number", type=int, help="output file parallelism", default=30)
    args = parser.parse_args()
    return args


def main(args):
    spark = SparkSession.builder.appName("[spark][resize dense table]").master("yarn").enableHiveSupport().getOrCreate()

    sc = spark.sparkContext
    output_bc_value = sc.broadcast(args.output)
    dense_file_rdd = (
        sc.wholeTextFiles(args.input)
        .map(lambda x: (x[0].split("/")[-1], x[0].split("/")[-2], x[1]))
        .flatMap(mapIndexToDenseRecord)
    )

    whole_data = dense_file_rdd.collect()
    res = process_whole_text(whole_data, args.number)

    res_rdd = sc.parallelize(res, args.number)
    res_rdd.foreachPartition(lambda p: write_dense_partition(p, output_bc_value))


if __name__ == "__main__":
    args = parse_args()
    main(args)
