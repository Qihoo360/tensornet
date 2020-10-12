#include "core/tools/graph.h"

#include "core/tools/graph_base.h"
#include "core/tools/object_pool.h"

#include "tensorflow/core/platform/logging.h"

#include <vector>
#include <map>
#include <set>
#include <iomanip>
#include <fstream>

#include <boost/algorithm/string.hpp>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tf_serving {

class ConfParser {
public:
    int Init(const std::string& file) {
        std::ifstream file_if(file);
        if (!file_if.is_open()) {
            LOG(ERROR) << "open " << file << " failed.";
            return -1;
        }

        std::string input;
        while(getline(file_if, input)) {
            if (input.front() == '#' || input.front() == '\0') {
                continue;
            }

            boost::trim(input);
            std::vector<std::string> items;
            boost::split(items, input, boost::is_any_of(":"));
            if (items.size() != 2) {
                continue;
            }

            for (auto& item : items) {
                boost::trim(item);
            }
            _table.insert({items[0], items[1]});
        }

        return 0;
    }

    int Get(const std::string& key, std::string& value) {
        if (_table.count(key) == 0) {
            LOG(ERROR) << "conf has no item:" << key;
            return -1;
        }
        value = _table[key];

        return 0;
    }

private:
    std::map<std::string, std::string> _table;
};

class GraphObj {
public:
    Graph* GetGraph() {
        return &_graph;
    }

    GraphObj(const std::string& path) {
        std::string var_params_file = path + "/dense_params.txt";
        std::ifstream var_params_if(var_params_file);
        if (!var_params_if.is_open()) {
            LOG(ERROR) << "open " << var_params_file << " failed.";
            return;
        }

        std::string input;
        while(getline(var_params_if, input)) {
            std::vector<std::string> var_params_vec;
            boost::algorithm::split(var_params_vec, input, boost::is_any_of("\t"));
            if (var_params_vec.size() != 2) {
                LOG(ERROR) << "var_params_file format error.";
                return;
            }
            std::string var_params_name = var_params_vec[0];
            boost::trim(var_params_name);
            std::vector<std::string> var_params_weight_strs;
            boost::algorithm::split(var_params_weight_strs, var_params_vec[1], boost::is_any_of("\1"));

            int index = _graph.LookupVariableIndex(var_params_name);
            size_t var_size = _graph.arg_size(index);

            //LOG(INFO) << "arg name:" << var_params_name <<", var size:" << var_size;
            if (var_size != var_params_weight_strs.size() * sizeof(float)) {
                LOG(ERROR) << "var param " << var_params_name << " size is " << var_params_weight_strs.size()
                           << ", expected " << var_size / sizeof(float);
                return;
            }
            std::vector<float> var_params_weight;
            for (auto& w : var_params_weight_strs) {
                var_params_weight.push_back(std::stof(w));
            }
            std::copy(var_params_weight.begin(), var_params_weight.end(), static_cast<float*>(_graph.arg_data(index)));

            //std::string weight_str = "";
            //for (size_t z = 0; z < var_params_weight.size(); ++z) {
            //    weight_str += "," + std::to_string((static_cast<float*>(_graph.arg_data(index))[z]));
            //}
        }
    }

private:
    Graph _graph;
};

class WideDeepGraph : public GraphBase {
public:
    WideDeepGraph() {}

    virtual ~WideDeepGraph() {
        if (_graph_pool != NULL) {
            delete _graph_pool;
            _graph_pool = NULL;
        }
    }

    // @param arg_type_info : feed_wide_emb, feed_deep_emb, feed_deep_lr_emb
    int Init(const char* conf_file) {
        ConfParser conf;
        if (conf.Init(conf_file) < 0) {
            LOG(ERROR) << "parse conf error:" << conf_file;
            return -1;
        }

        std::string max_batch_size;
        if (conf.Get("max_batch_size", max_batch_size) < 0) {
            LOG(ERROR) << "conf get item max_batch_size failed.";
            return -1;
        }

        std::string slot_num_str;
        int slot_num = 0;
        if (conf.Get("slot_num", slot_num_str) < 0) {
            LOG(ERROR) << "conf get item slot_num failed.";
            return -1;
        }

        std::string pool_size_str;
        int pool_size = 1;
        if (conf.Get("pool_size", pool_size_str) < 0) {
            LOG(ERROR) << "conf get item pool_size failed.";
            return -1;
        }

        try {
            _max_batch_size = std::stoi(max_batch_size);
            slot_num = std::stoi(slot_num_str);
            pool_size = std::stoi(pool_size_str);
        } catch(...) {
            LOG(ERROR) << "conf max_batch_size or slot_num or pool_size is not int.";
            return -1;
        }

        std::string arg_type_info;
        if (conf.Get("arg_type_info", arg_type_info) < 0) {
            LOG(ERROR) << "conf get item arg_type_info failed.";
            return -1;
        }
        std::vector<std::string> arg_types_vec;
        boost::split(arg_types_vec, arg_type_info, boost::is_any_of(","));
        for (auto& a : arg_types_vec) {
            boost::trim(a);
        }

        std::string path;
        if (conf.Get("path", path) < 0) {
            LOG(ERROR) << "conf get item path failed.";
        }

        std::string feed_params_file = path + "/feed_params.txt";
        std::ifstream feed_params_if(feed_params_file);
        if (!feed_params_if.is_open()) {
            LOG(ERROR) << "open " << feed_params_file << " failed.";
            return -1;
        }

        std::string input;
        getline(feed_params_if, input);
        boost::trim(input);
        std::vector<std::string> feed_names;
        boost::split(feed_names, input, boost::is_any_of(","));

        Graph graph;

        std::map<std::string, int> feed_names_index_map;
        std::set<std::string> arg_types_set;
        for (auto& s : feed_names) {
            int index = graph.LookupArgIndex(s);
            feed_names_index_map.insert({s, index});

            auto ret = arg_types_set.insert(s.substr(0, s.rfind("_")));
            if (ret.second) {
                _emb_dims.push_back(graph.arg_size(index) / _max_batch_size / sizeof(float));
            }
        }

        if (arg_types_vec.size() != arg_types_set.size()) {
            LOG(ERROR) << "arg_types not match.";
            return -1;
        }
        _arg_type_num = arg_types_set.size();

        for (int i = 0; i < slot_num; ++i) {
            for (int t = 0; t < _arg_type_num; ++t) {
                std::string name = arg_types_vec[t] + "_" + std::to_string(i);
                if (feed_names_index_map.count(name) == 0) {
                    LOG(ERROR) << "Init Error, no " << name << " in feed_names.";
                    return -1;
                }
                _slot_index.push_back(feed_names_index_map[name]);
            }
        }

        _graph_pool = new ObjectPool<GraphObj>(pool_size, path);

        return 0;
    }

    int Run(const std::vector<TFFeaValues>& input, std::vector<float>& output) {
        output.clear();
        output.assign(input.size(), 0.0);

        int cursor = 0;
        int last_cursor = 0;

        GraphObj* graph_obj = GetGraphObj();
        if (graph_obj == NULL) {
            LOG(ERROR) << "GetGraphObj returns NULL ptr.";
            return -1;
        }

        Graph* graph = graph_obj->GetGraph();
        if (graph == NULL) {
            LOG(ERROR) << "GetGraph returns NULL ptr.";
            return -1;
        }

        for (size_t i = 0; i < input.size(); ++i) {
            cursor++;
            for (size_t s = 0; s < input[i].size(); ++s) {
                if (input[i][s].fm_array.size() != EmbSize()) {
                    LOG(ERROR) << "input format error, input size is " << input[i][s].fm_array.size()
                               << ", expected size is " << EmbSize();
                    return -1;
                }

                int start = 0;
                for (int t = 0; t < _arg_type_num; ++t) {
                    int dim = GetDim(t);
                    if (dim < 0) {
                        LOG(ERROR) << "GetDim error.";
                        return -1;
                    }
                    int end = start + dim;

                    int index = GetIndex(s, t);
                    //std::cerr << "index:" << index << std::endl;
                    if (index < 0) {
                        LOG(ERROR) << "GetIndex error.";
                        return -1;
                    }

                    std::copy(input[i][s].fm_array.data() + start, input[i][s].fm_array.data() + end,
                              static_cast<float*>(graph->arg_data(index)) + ((cursor - last_cursor - 1) * dim));

                    start = end;
                }
            }

            if (cursor % _max_batch_size == 0) {
                auto ok = graph->Run();
                if (!ok) {
                    LOG(ERROR) << "graph run failed.";
                    return -1;
                }

                std::copy(graph->result0_data(),
                          graph->result0_data() + cursor - last_cursor,
                          output.data() + last_cursor);
                last_cursor = cursor;
            }
        }

        if (last_cursor != cursor) {
            auto ok = graph->Run();
            if (!ok) {
                LOG(ERROR) << "graph run failed.";
                return -1;
            }

            std::copy(graph->result0_data(),
                      graph->result0_data() + cursor - last_cursor,
                      output.data() + last_cursor);
        }

        if (cursor != output.size()) {
            LOG(ERROR) << "graph run output size not correct.";
            return -1;
        }


        ReturnGraph(graph_obj);

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
            LOG(ERROR) << "GetIndex error: slot_index is " << slot_index
                       << ", arg_type is " << arg_type
                       << ", max index is " << _slot_index.size();
            return -1;
        }
        return _slot_index[index];
    }

    int GetDim(int arg_type) {
        if (arg_type >= _slot_index.size() || arg_type < 0) {
            LOG(ERROR) << "GetDim error: arg_type is " << arg_type
                       << ", max arg_type is " << _slot_index.size();
            return -1;
        }
        return _emb_dims[arg_type];
    }

    GraphObj* GetGraphObj() {
        GraphObj* obj = _graph_pool->GetObject();
        if (obj == NULL) {
            LOG(ERROR) << "GetGraph failed.";
            return NULL;
        }

        return obj;
    }

    void ReturnGraph(GraphObj* obj) {
        _graph_pool->ReturnObject(obj);
    }

private:
    int _max_batch_size;
    int _arg_type_num;
    std::vector<int> _emb_dims;
    std::vector<int> _slot_index;
    ObjectPool<GraphObj>* _graph_pool;
};

}; // namespace tf_serving

extern "C" void* CreateInstance() {
    tf_serving::WideDeepGraph* obj = new tf_serving::WideDeepGraph();
    if (obj == nullptr) {
        LOG(ERROR) << "CreateInstance Error.";
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

