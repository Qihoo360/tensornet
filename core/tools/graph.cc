#include "core/tools/graph.h"

#include <vector>
#include <map>
#include <set>
#include <iomanip>

#include <boost/algorithm/string.hpp>

#include "core/tools/object_pool.h"
#include "core/tools/dual_dict.h"

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

struct TFFeaValue {
    TFFeaValue() : slot(0) {}

    int slot;
    std::vector<float> fm_array;
};

class GraphObj {
public:
    Graph* GetGraph() {
        return &_graph;
    }

    uint64_t Revision() const {
        return _revision;
    }

    int Init(const std::string& path) {
        std::string revision_file = path + "/revision.txt";
        std::ifstream revision_if(revision_file);
        if (!revision_if.is_open()) {
            std::cerr << "open " << revision_file << " failed." << std::endl;
            return -1;
        }

        std::string input;
        getline(revision_if, input);
        boost::trim(input);
        _revision = std::stoull(input);
      
        std::string var_params_file = path + "/dense_params.txt";
        std::ifstream var_params_if(var_params_file);
        if (!var_params_if.is_open()) {
            std::cerr << "open " << var_params_file << " failed." << std::endl;
            return -1;
        }

        while(getline(var_params_if, input)) {
            std::vector<std::string> var_params_vec;
            boost::algorithm::split(var_params_vec, input, boost::is_any_of("\t"));
            if (var_params_vec.size() != 2) {
                std::cerr << "var_params_file format error." << std::endl;
                return -1;
            }
            std::string var_params_name = var_params_vec[0];
            boost::trim(var_params_name);
            std::vector<std::string> var_params_weight_strs;
            boost::algorithm::split(var_params_weight_strs, var_params_vec[1], boost::is_any_of("\1"));

            int index = _graph.LookupVariableIndex(var_params_name);
            size_t var_size = _graph.arg_size(index);
            if (var_size != var_params_weight_strs.size() * sizeof(float)) {
                std::cerr << "var param " << var_params_name << " size is " << var_params_weight_strs.size()
                          << ", expected " << var_size / sizeof(float) 
                          << std::endl;
                return -1;
            }
            std::vector<float> var_params_weight;
            for (auto& w : var_params_weight_strs) {
                var_params_weight.push_back(std::stof(w));
            }
            std::copy(var_params_weight.begin(), var_params_weight.end(), static_cast<float*>(_graph.arg_data(index)));
        }

        return 0;
    }

private:
    Graph _graph;
    uint64_t _revision = 0;
};

class GraphDict {
public:
    GraphDict(const std::string& path, const std::string& done_file) {
        _graph_dict.Init(path, done_file);
    }

    Graph* GetGraph() {
        return _graph_dict.GetPreferDict()->GetGraph();
    }

private:
    hsd::DualDict<GraphObj> _graph_dict;
};

typedef std::vector<TFFeaValue> TFFeaValues;

class GraphManager {
public:
    static GraphManager* Instance() {
        static GraphManager graph_manager;
        return &graph_manager;
    }

    // @param arg_type_info : feed_wide_emb, feed_deep_emb, feed_deep_lr_emb
    int Init(int slot_num, int max_batch_size, std::string arg_type_info, 
             const std::string& path, const std::string& done_file) {
        _max_batch_size = max_batch_size;

        std::vector<std::string> arg_types_vec;
        boost::split(arg_types_vec, arg_type_info, boost::is_any_of(","));
        for (auto& a : arg_types_vec) {
            boost::trim(a);
        }

        std::string feed_params_file = path + "/feed_params.txt";
        std::ifstream feed_params_if(feed_params_file);
        if (!feed_params_if.is_open()) {
            std::cerr << "open " << feed_params_file << " failed." << std::endl;
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
            std::cerr << "arg_types not match." << std::endl;
            return -1;
        }
        _arg_type_num = arg_types_set.size();

        for (int i = 0; i < slot_num; ++i) {
            for (int t = 0; t < _arg_type_num; ++t) {
                std::string name = arg_types_vec[t] + "_" + std::to_string(i);
                if (feed_names_index_map.count(name) == 0) {
                    std::cerr << "Init Error, no " << name << " in k_names." << std::endl;
                    return -1;
                }
                _slot_index.push_back(feed_names_index_map[name]);
            }
        }

        size_t pool_size = 5;
        _graph_pool = new ObjectPool<GraphDict>(pool_size, path, done_file);

        return 0;
    }
    
    int MaxBatchSize() {
        return _max_batch_size;
    }

    int ArgTypeNum() {
        return _arg_type_num;
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

    GraphDict* GetGraph() {
        GraphDict* obj = _graph_pool->GetObject();
        if (obj == NULL) {
            std::cerr << "GetGraph failed." << std::endl;
            return NULL;
        }

        return obj;
    }

    void ReturnGraph(GraphDict* obj) {
        _graph_pool->ReturnObject(obj);
    }

private:
    GraphManager() {}
    ~GraphManager() {}

private:
    int _max_batch_size;
    int _arg_type_num;
    std::vector<int> _emb_dims;
    std::vector<int> _slot_index;
    ObjectPool<GraphDict>* _graph_pool;
};

extern "C" int Init(int slot_num, int max_batch_size, 
                    const char* arg_type_info, int arg_type_info_len, 
                    const char* path, int path_len, 
                    const char* done_file, int done_file_len) {
    return GraphManager::Instance()->Init(slot_num, max_batch_size, 
                                          std::string(arg_type_info, arg_type_info_len), 
                                          std::string(path, path_len), 
                                          std::string(done_file, done_file_len));
}

extern "C" int Run(const std::vector<TFFeaValues>& input,
                   std::vector<float>& output) {
    output.clear();
    output.assign(input.size(), 0.0);

    int cursor = 0;
    int last_cursor = 0;

    GraphDict* graph_dict = GraphManager::Instance()->GetGraph();
    if (graph_dict == NULL) {
        std::cerr << "graph dict is NULL." << std::endl;
        return -1;
    }

    Graph* graph = graph_dict->GetGraph();
    if (graph == NULL) {
        std::cerr << "graph is NULL." << std::endl;
        return -1;
    }
    

    for (size_t i = 0; i < input.size(); ++i) {
        cursor++;
        for (size_t s = 0; s < input[i].size(); ++s) {
            if (input[i][s].fm_array.size() != GraphManager::Instance()->EmbSize()) {
                std::cerr << "input format error, input size is " << input[i][s].fm_array.size() 
                          << ", expected size is " << GraphManager::Instance()->EmbSize()
                          << std::endl;
                return -1;
            }

            int start = 0;
            for (int t = 0; t < GraphManager::Instance()->ArgTypeNum(); ++t) {
                int dim = GraphManager::Instance()->GetDim(t);
                if (dim < 0) {
                    std::cerr << "GraphManager::Instance()->GetDim error." << std::endl;
                    return -1;
                }
                int end = start + dim;

                int index = GraphManager::Instance()->GetIndex(s, t);
                if (index < 0) {
                    std::cerr << "GraphManager::Instance()->GetIndex error." << std::endl;
                    return -1;
                }

                std::copy(input[i][s].fm_array.data() + start, input[i][s].fm_array.data() + end, 
                          static_cast<float*>(graph->arg_data(index)) + ((cursor - last_cursor - 1) * dim));

                start = end;
            }
        }

        if (cursor % GraphManager::Instance()->MaxBatchSize() == 0) {
            auto ok = graph->Run();
            if (!ok) {
                std::cerr << "graph run failed." << std::endl;
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
            std::cerr << "graph run failed." << std::endl;
            return -1;
        }

        std::copy(graph->result0_data(),
                  graph->result0_data() + cursor - last_cursor,
                  output.data() + last_cursor);
    }

    if (cursor != output.size()) {
        std::cerr << "graph run output size not correct." << std::endl;
        return -1;
    }

    GraphManager::Instance()->ReturnGraph(graph_dict);

    return 0;
}

