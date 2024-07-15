#coding=utf-8
import gzip
import struct
import io
from struct import unpack,pack
from io import BytesIO
import hdfs3
from hdfs3 import HDFileSystem
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
import re
import math


hdfs_head_pattern = r"hdfs://[^/]+"
sparse_ada_grad_schema = ['sign', 'dim', 'weights', 'g2sum', 'show', 'no_show_days']
BLOCK_NUM = 8


def get_hdfs_head(path):
    hdfs_head = "hdfs://ss-hadoop2"
    match_res = re.match(hdfs_head_pattern, path)
    if match_res:
        return match_res.group(0)
    return hdfs_head


def get_handle_name(path):
    """
    get handle name from file path
    File Path should be prefix/sparse_table/handle_name/rank_num/file.gz
    """
    elements = path.split('/')
    return elements[-3] if elements else None


def get_sign_partition_key(sign, mod):
    block_id = sparse_key_hasher(int(sign))
    sign_mod = int(sign) % int(mod)
    return block_id * int(mod) + sign_mod


def sparse_key_hasher(sign):  
    flipped_sign = (int(sign) >> 32) | (int(sign) << 32)  
    return flipped_sign % BLOCK_NUM


def process_txt_line(line):
    """
    Fetch sign and weights from sparse
    Data should be seperated by '\t', sign\tdim_num\tdim_num*weight
    """
    data_list = line.split('\t')
    if len(data_list) < 3:
        return ("", 0, [], "", "", 0)
    else:
        sign = data_list[0]
        dim = int(data_list[1])
        weights = data_list[2: dim+2]
        g2sum = data_list[dim+2]
        show = data_list[dim+3]
        no_show_days = 0
        if len(data_list) > dim + 4:
            no_show_days = int(data_list[dim+4])
    return (sign, dim, weights, g2sum, show, no_show_days)


def fetch_hanlds(input_path):
    hdfs_head = get_hdfs_head(input_path)
    hdfs = HDFileSystem(host=hdfs_head)
    res_file = hdfs.glob(input_path)[0]
    while hdfs.info(res_file)['kind'] != 'file':
        res_file = hdfs.ls(res_file)[0]
    sparse_parent = "/".join(res_file.split("/")[:-3])
    return [ path.split('/')[-1] for path in hdfs.ls(sparse_parent)]


def appendIndex(index, iterator):
    for row in iterator:
        yield (index, row)


def process_txt_whole_line(line):
    """
    Fetch sign from sparse, and return whole line
    """
    data_list = line.split('\t')
    if len(data_list) < 3:
        return ("", "")
    else:
        sign = data_list[0]
    return (sign, line)


def resize_partition(iterator, bc_output_path, bc_format, bc_number, bc_handle_names):
    output_path = bc_output_path.value
    file_format = bc_format.value
    total_rank_number = bc_number.value
    handle_names = bc_handle_names.value
    hdfs_head = get_hdfs_head(output_path)
    hdfs = HDFileSystem(host=hdfs_head)
    handle_io_map = {}
    par_index = None
    rank_num = None
    block_id = None

    for row in iterator:
        if not par_index:
            par_index = int(row[0])
            rank_num = par_index % total_rank_number
            block_id = int(par_index / total_rank_number)
        data_row = row[1]
        key = data_row[0]
        dim = int(data_row[1])
        handle = data_row[6]
        if handle not in handle_io_map:
            handle_io_map[handle] = init_sparse_file(dim, file_format)
        if file_format == 'txt':
            write_txt_data(data_row, handle_io_map[handle][1])
        else:
            write_binary_data(data_row, handle_io_map[handle][1])

    for handle in handle_names:
        if handle not in handle_io_map:
            handle_io_map[handle] = init_sparse_file(8, file_format)

        hdfs.makedirs('{}/{}/rank_{}'.format(output_path, handle, rank_num), 0o755)
        handle_io_map[handle][1].close()
        file_path = '{}/{}/rank_{}/sparse_block_{}_{}.gz'.format(output_path, handle, rank_num, block_id, file_format)
        if hdfs.exists(file_path):
            hdfs.rm(file_path)
        with hdfs.open(file_path,'wb') as f:
            f.write(handle_io_map[handle][0].getvalue())


def init_sparse_file(dim, file_format):
    compressed_data = io.BytesIO()
    gzip_io = gzip.GzipFile(fileobj=compressed_data, mode='wb')
    if file_format == 'txt':
        gzip_io.write("opt_name:AdaGrad\n".encode('utf-8'))
        gzip_io.write(("dim:{}\n".format(dim)).encode('utf-8'))
    else:
        gzip_io.write(struct.pack('i', int(dim)))
    return (compressed_data, gzip_io)


def write_txt_data(input_data, file_io):
    result_list = list(input_data[:2]) + input_data[2] + list(input_data[3:6])
    res_data = '\t'.join(str(item) for item in result_list) 
    file_io.write((res_data + '\n').encode('utf-8'))


def write_binary_data(input_data, file_io):
    file_io.write(struct.pack('Q', int(input_data[0])))
    file_io.write(struct.pack('i', int(input_data[1])))
    for weight in input_data[2]:
        file_io.write(struct.pack('f', float(weight)))
    file_io.write(struct.pack('f', float(input_data[3])))
    file_io.write(struct.pack('f', int(input_data[4])))
    file_io.write(struct.pack('i', int(input_data[5])))


def process_binary_partition(iterator):
    """
    Used by mapPartition to convert binary file to line record
    File has an int for dim_num on top, then each data should be sign, dim_num * weight, g2sum, show, no_show_days
    """
    for filename, file_content in iterator:
        handle = filename.split('/')[-3]
        with io.BytesIO(file_content) as fc:
            with gzip.open(fc, 'rb') as gzip_file:
                dim = unpack('i', gzip_file.read(4))[0]
                while True:
                    try:
                        long_value = unpack('Q', gzip_file.read(8))[0]
                        sign_str = str(long_value)
                        weights = [unpack('f', gzip_file.read(4))[0] for _ in range(8)]
                        g2sum = unpack('f', gzip_file.read(4))[0]
                        show_rate = unpack('f', gzip_file.read(4))[0]
                        no_show_days = unpack('i', gzip_file.read(4))[0]
                        yield (sign_str, dim, weights, str(g2sum), str(show_rate), no_show_days, handle)
                    except Exception as e:
                        print(e)
                        yield ("", 0, [], "", "", 0, "")
                        break


def output_line(line, need_bracket):
    key = line[0]
    dims = [ element for element in line[1] ]
    if need_bracket:
        output_list = [key, "["] + dims + ["]"]
    else:
        output_list = [key] + dims
    return "\t".join(str(item) for item in output_list)


def load_sparse_table_to_df(sc, input_path, file_format):
    """
    load sparse table file to DF. output format should be ['sign', 'dim', 'weights', 'g2sum', 'show', 'no_show_days', 'handle'] 
    """
    if file_format == 'txt':
        get_handle_name_udf = udf(get_handle_name, StringType())
        dims_df = sc.textFile(input_path)\
                     .map(lambda x: process_txt_line(x))\
                     .toDF(sparse_ada_grad_schema)\
                     .withColumn("input_file_name",F.input_file_name())\
                     .withColumn("handle", get_handle_name_udf(col("input_file_name")))\
                     .drop("input_file_name")\
                     .filter(col("sign") != "")
        return dims_df
    else:
        dims_df = sc.binaryFiles(input_path)\
                     .mapPartitions(process_binary_partition)\
                     .toDF(sparse_ada_grad_schema + ['handle']).filter(col("sign") != "")
        return dims_df
   

def mapIndexToDenseRecord(row):
    """
    map file name, handle name, and append index to line
    """
    file_index = int(row[0])
    handle_name = row[1]
    whole_text = row[2]
    lines = whole_text.split("\n")
    data_lines = [ line for line in lines if len(line.split("\t")) > 2]
    data_lines_with_index = [ (handle_name, file_index, line_index, line) for line_index, line in enumerate(data_lines)]
    return data_lines_with_index


def process_whole_text(whole_data, number):
    whole_data.sort(key=lambda x:(x[0], x[1], x[2]))
    handle_map = {}
    for ele in whole_data:
        if ele[0] in handle_map:
            handle_map[ele[0]].append(ele)
        else:
            handle_map[ele[0]] = [ele]
    res = []
    for handle in handle_map:
        handle_list = handle_map[handle]
        part_num = number
        total_num = len(handle_list)
        each_part_num = math.ceil(total_num / part_num)
        start_index = 0
        for i in range(part_num):
            data_line_txt = "total_num:{}\nrank_num:{}\n".format(total_num, part_num)
            if start_index + each_part_num > total_num:
                end_index = total_num
            else:
                end_index = start_index + each_part_num
            sub_list = handle_list[start_index : end_index]
            sub_len = len(sub_list)
            sub_part_num = math.ceil(sub_len / BLOCK_NUM)
            sub_start_index = 0
            for j in range(BLOCK_NUM):
                if sub_start_index + sub_part_num > sub_len:
                    sub_end_index = sub_len
                    array_size = sub_len - sub_start_index
                else:
                    sub_end_index = sub_start_index + sub_part_num
                    array_size = sub_part_num
                data_line_txt += "opt_name:Adam\narray_size:{}\nbeta1_power:0\nbeta1_power:0\n".format(array_size)
                for sub_line in sub_list[sub_start_index:sub_end_index]:
                    data_line_txt += sub_line[3] + '\n'
                data_line_txt += "\n\n"
                sub_start_index = sub_end_index
            start_index = end_index
            res.append((i, (handle, data_line_txt)))
    return res


def write_dense_partition(iterator, bc_output_path):
    output_path = bc_output_path.value
    hdfs_head = get_hdfs_head(output_path)
    hdfs = HDFileSystem(host=hdfs_head)
    for row in iterator:
        file_index = int(row[0])
        handle_name = row[1][0]
        data = row[1][1]
        compressed_data = io.BytesIO()
        compressed_data.write(data.encode('utf-8'))
        hdfs.makedirs('{}/{}'.format(output_path, handle_name), 0o755)
        with hdfs.open('{}/{}/{}'.format(output_path, handle_name, file_index),'wb') as f:
            f.write(compressed_data.getvalue())
