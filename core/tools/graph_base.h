#ifndef GRAPH_BASE_H_
#define GRAPH_BASE_H_

#include <vector>

namespace tf_serving {

struct TFFeaValue {
    TFFeaValue() : slot(0) {}

    int slot;
    std::vector<float> fm_array;
};

typedef std::vector<TFFeaValue> TFFeaValues;

class GraphBase {
public:
    GraphBase() {}

    virtual ~GraphBase() {}

    // wide_deep_graph.cc
    //virtual int Init(int slot_num, int max_batch_size, const char* arg_type_info, int length) = 0;

    // wide_deep_graph_set.cc
    virtual int Init(const char* conf) = 0;

    virtual int Run(const std::vector<TFFeaValues>& input, std::vector<float>& output) = 0;
};

};

#endif  // GRAPH_BASE_H_
