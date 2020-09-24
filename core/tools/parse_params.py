import sys
import argparse

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.lib.io import file_io
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util

def parse_args():
    parser = argparse.ArgumentParser(description='parse dense params args')
    parser.add_argument(
        "--input_saved_model_dir",
        type=str,
        help="TensorFlow SavedModel dir to load.")
    parser.add_argument(
        "--saved_model_tags",
        type=str,
        default="serve",
        help="SavedModel tags.")
    parser.add_argument(
        "--output_node_names",
        type=str,
        default="StatefulPartitionedCall",
        help="SavedModel output node names.")
    parser.add_argument(
        "--output_dense_params_file",
        type=str,
        help="Output dense params file.")
    parser.add_argument(
        "--output_graph_input_names_file",
        type=str,
        help="Output graph input names file.")
    return parser.parse_args()

def parse_dense_params(input_saved_model_dir,
                       saved_model_tags,
                       output_node_names,
                       output_dense_params_file):
    input_graph_filename = None
    input_saver_def_path = False
    input_binary = False
    checkpoint_path = None
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    output_graph_filename = "./freezed_graph_def.pb"

    freeze_graph.freeze_graph(input_graph_filename,
                              input_saver_def_path,
                              input_binary,
                              checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_filename,
                              clear_devices,
                              "",
                              "",
                              "",
                              input_meta_graph,
                              input_saved_model_dir,
                              saved_model_tags)
    
    with file_io.FileIO(output_graph_filename, mode="rb") as read_f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(read_f.read())
        with file_io.FileIO(output_dense_params_file, mode="w") as write_f:
            # filt not need case
            try:
                for n in graph_def.node:
                    variable_name = "param_{}".format(n.name.replace('/', '_'))
                    dense_params = "\1".join(str(i) for i in tensor_util.MakeNdarray(n.attr['value'].tensor).flatten().tolist())
                    out_line = variable_name + "\t" + dense_params + "\n"
                    write_f.write(out_line)
            except:
                pass
    
def parse_graph_input_names(input_saved_model_dir,
                            saved_model_tags,
                            output_graph_input_names_file):
    saved_model_def = saved_model_utils.get_meta_graph_def(input_saved_model_dir, saved_model_tags)
    inputs = saved_model_def.signature_def['serving_default'].inputs
    with file_io.FileIO(output_graph_input_names_file, "w") as write_f:
        write_f.write(",".join([ "feed_" + name for name in inputs.keys() ]))

def main():
    args = parse_args()

    parse_dense_params(input_saved_model_dir=args.input_saved_model_dir,
                       saved_model_tags=args.saved_model_tags,
                       output_node_names=args.output_node_names,
                       output_dense_params_file=args.output_dense_params_file)

    parse_graph_input_names(input_saved_model_dir=args.input_saved_model_dir,
                            saved_model_tags=args.saved_model_tags,
                            output_graph_input_names_file=args.output_graph_input_names_file)

if __name__ == "__main__":
    main()
