#coding=utf-8
import gzip
import struct
import io
from struct import unpack
from io import BytesIO

def get_handle_name(path):
    """
    get handle name from file path
    File Path should be prefix/sparse_table/handle_name/rank_num/file.gz
    """
    elements = path.split('/')
    return elements[-3] if elements else None

def process_txt_line(line):
    """
    Fetch sign and weights from sparse
    Data should be seperated by '\t', sign\tdim_num\tdim_num*weight
    """
    data_list = line.split('\t')
    if len(data_list) < 3:
        return ("", [])
    else:
        sign = data_list[0]
        dim = int(data_list[1])
        weights = data_list[2: dim+2]
    return (sign, weights)


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
                        yield (handle, sign_str, weights)
                    except Exception as e:
                        print(e)
                        yield ("","",[])
                        break


def output_line(line, need_bracket):
    key = line[0]
    dims = [ element for element in line[1] ]
    if need_bracket:
        output_list = [key, "["] + dims + ["]"]
    else:
        output_list = [key] + dims
    return "\t".join(str(item) for item in output_list)
