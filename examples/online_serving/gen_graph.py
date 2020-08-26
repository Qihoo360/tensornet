import os, sys
from string import Template

SLOT_NUM = int(sys.argv[1])
if SLOT_NUM % 2 != 0:
    sys.stderr.write("SLOT_NUM is not times of 2")
    exit(-1)

SLOT_NUM = int(SLOT_NUM / 2)

def string_template():
    code = Template("""
        std::copy(input[i][${slot}].data(), input[i][${slot}].data() + dim[0], graph.arg_feed_wide_emb_${slot}_data() + i * dim[0]);
        std::copy(input[i][${slot}].data() + dim[0], input[i][${slot}].data() + dim[0] + dim[1], graph.arg_feed_deep_emb_${slot}_data() + i * dim[1]);""")
    for slot in range(SLOT_NUM):
        code_str = code.safe_substitute(slot=slot)
        print(code_str)

def gen_code():
    print("""// This file is MACHINE GENERATED! Do not edit.
// Source file is gen_graph.py

#include "examples/online_serving/graph.h"

#include <vector>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

extern "C" int Run(const std::vector<std::vector<std::vector<float>>>& input,
                   std::vector<float>& output) {
    Eigen::ThreadPool tp(std::thread::hardware_concurrency());
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
    Graph graph;
    graph.set_thread_pool(&device);

    std::vector<int> dim = {1, 8};
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i].size() != Graph::kNumArgs / 2) {
            std::cerr << "TFFeaValues size is wrong, expected " << Graph::kNumArgs / 2 << " but get" << input[i].size() << std::endl;
            return -1;
        }
        if (input[i][0].size() != dim[0] + dim[1]) {
            std::cerr << "embedding size is wrong, expected " << dim[0] + dim[1] << " but get" << input[i][0].size() << std::endl;
            return -1;
        }
    """)
    string_template()
    print("""
    }

    auto ok = graph.Run();
    if (!ok) {
        return -1;
    }

    output.clear();
    output.assign(input.size(), 0.0);
    std::copy(graph.result0_data(), graph.result0_data() + input.size(), output.data());

    return 0;
}

    """)
    
gen_code()

