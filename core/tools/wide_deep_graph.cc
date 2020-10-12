#include "core/tools/graph.h"

#include "core/tools/graph_base.h"

#include <vector>
#include <map>
#include <set>
#include <iomanip>

#include <boost/algorithm/string.hpp>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tf_serving {

class WideDeepGraph : public GraphBase {
public:
    WideDeepGraph() {}

    virtual ~WideDeepGraph() {}

    // @param arg_type_info : feed_wide_emb, feed_deep_emb, feed_deep_lr_emb
    int Init(int slot_num, int max_batch_size, const char* arg_type_info, int length) {
        _max_batch_size = max_batch_size;

        std::vector<std::string> arg_types_vec;
        boost::split(arg_types_vec, std::string(arg_type_info, length),  boost::is_any_of(","));
        for (auto& a : arg_types_vec) {
            boost::trim(a);
        }

        Graph graph;

        std::map<std::string, int> k_names_map;
        const char** names = Graph::StaticArgNames();
        for(size_t i = 0; i < Graph::kNumArgs; ++i) {
            if (names[i] == nullptr) {
                break;
            }
            int index = graph.LookupArgIndex(names[i]);
            k_names_map[names[i]] = index;
        }

        std::set<std::string> arg_types_set;
        for (auto& s : k_names_map) {
            arg_types_set.insert(s.first.substr(0, s.first.rfind("_")));
        }

        if (arg_types_vec.size() != arg_types_set.size()) {
            std::cerr << "arg_types not match." << std::endl;
            return -1;
        }
        _arg_type_num = arg_types_set.size();

        bool emb_size_inited = false;

        for (int i = 0; i < slot_num; ++i) {
            for (int t = 0; t < _arg_type_num; ++t) {
                std::string name = arg_types_vec[t] + "_" + std::to_string(i);
                if (k_names_map.count(name) == 0) {
                    std::cerr << "Init Error, no " << name << " in k_names." << std::endl;
                    return -1;
                }
                _slot_index.push_back(k_names_map[name]);
                if (!emb_size_inited) {
                    _emb_dims.push_back(graph.arg_size(k_names_map[name]) / _max_batch_size / sizeof(float));
                }
            }
            emb_size_inited = true;
        }

        return 0;
    }

    int Run(const std::vector<TFFeaValues>& input, std::vector<float>& output) {
        output.clear();
        output.assign(input.size(), 0.0);

        int cursor = 0;
        int last_cursor = 0;

        const char** names = Graph::StaticArgNames();
        std::cerr << "name:" << names[0] << "," << names[1] << "," << names[2] << std::endl;
        Graph graph;
        for (size_t i = 0; i < input.size(); ++i) {
            cursor++;
            for (size_t s = 0; s < input[i].size(); ++s) {
                if (input[i][s].fm_array.size() != EmbSize()) {
                    std::cerr << "input format error, input size is " << input[i][s].fm_array.size() 
                            << ", expected size is " << EmbSize()
                            << std::endl;
                    return -1;
                }

                std::string weight_str = "";
                for (size_t k = 0; k < input[i][s].fm_array.size(); ++k) {
                    weight_str += "," + std::to_string(input[i][s].fm_array[k]);
                }
                std::cerr << "weight:" << weight_str << std::endl;

                int start = 0;
                for (int t = 0; t < _arg_type_num; ++t) {
                    int dim = GetDim(t);
                    if (dim < 0) {
                        std::cerr << "GetDim error." << std::endl;
                        return -1;
                    }
                    int end = start + dim;

                    int index = GetIndex(s, t);
                    if (index < 0) {
                        std::cerr << "GetIndex error." << std::endl;
                        return -1;
                    }
                    std::copy(input[i][s].fm_array.data() + start, input[i][s].fm_array.data() + end,
                              static_cast<float*>(graph.arg_data(index)) + ((cursor - last_cursor - 1) * dim));

                    start = end;
                    //std::string weight_str = "";
                    //for (int k = 0; k < dim; ++k)  {
                    //    weight_str += "," + std::to_string((static_cast<float*>(graph.arg_data(index)) + ((cursor - last_cursor - 1) * dim))[k]);
                    //}
                    //std::cerr << "weight_str:" << weight_str << std::endl;
                }
            }

            if (cursor % _max_batch_size == 0) {
                auto ok = graph.Run();
                if (!ok) {
                    std::cerr << "graph run failed." << std::endl;
                    return -1;
                }

                std::copy(graph.result0_data(),
                          graph.result0_data() + cursor - last_cursor,
                          output.data() + last_cursor);
                last_cursor = cursor;
            }
        }

        if (last_cursor != cursor) {
            auto ok = graph.Run();
            if (!ok) {
                std::cerr << "graph run failed." << std::endl;
                return -1;
            }

            std::copy(graph.result0_data(),
                      graph.result0_data() + cursor - last_cursor,
                      output.data() + last_cursor);
        }

        if (cursor != output.size()) {
            std::cerr << "graph run output size not correct." << std::endl;
            return -1;
        }

        return 0;
    }

    int EmbSize() {
        int emb_size = 0;
        for (auto d : _emb_dims) {
            emb_size += d;
        }
        return emb_size;
    }

    int GetIndex(int slot_index, int arg_type) {
        size_t index = slot_index *  _arg_type_num + arg_type;
        if (index >= _slot_index.size() || index < 0) {
            std::cerr << "GetIndex error: slot_index is " << slot_index
                      << ", arg_type is " << arg_type
                      << ", max index is " << _slot_index.size()
                      << std::endl;
            return -1;
        }
        return _slot_index[index];
    }

    int GetDim(int arg_type) {
        if (arg_type >= _slot_index.size() || arg_type < 0) {
            std::cerr << "GetDim error: arg_type is " << arg_type
                      << ", max arg_type is " << _slot_index.size()
                      << std::endl;
            return -1;
        }
        return _emb_dims[arg_type];
    }

private:
    int _max_batch_size;
    int _arg_type_num;
    std::vector<int> _emb_dims;
    std::vector<int> _slot_index;
};

};

extern "C" void* CreateInstance() {
    tf_serving::WideDeepGraph* obj = new tf_serving::WideDeepGraph();
    if (obj == nullptr) {
        std::cerr << "CreateInstance Error." << std::endl;
        return nullptr;
    }
    return reinterpret_cast<void*>(obj);

    return 0;
}

extern "C" void DestroyInstance(void* instance) {
    tf_serving::WideDeepGraph* obj = reinterpret_cast<tf_serving::WideDeepGraph*>(instance);
    if (obj != nullptr) {
        delete obj;
        obj = nullptr;
    }
}

