// Copyright 2020-2025 Qihoo Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "core/utility/random.h"

using namespace std::chrono;

typedef int (*RUN_FUNC)(const std::vector<std::vector<std::vector<float>>>&, std::vector<float>&);

const int k_batch_size = 32;

void InitWeight(int dim, std::vector<float>& weight) {
    weight.clear();
    auto& reng = tensornet::local_random_engine();
    auto distribution = std::normal_distribution<float>(0, 1 / sqrt(dim));

    for (int i = 0; i < dim; ++i) {
        weight.push_back(distribution(reng) * 0.001);
    }
}

int combine_fea(std::vector<std::vector<float>> emb_feas, std::vector<float>& merged_feas) {
    if (emb_feas.size() % 2 != 0) {
        std::cerr << "combine_fea error." << std::endl;
        return -1;
    }
    float wide_lr = 0.0;
    std::vector<float> dnn_vec(8, 0.0);
    for (size_t i = 0; i < emb_feas.size(); i++) {
        if (i % 2 == 0) {
            wide_lr += emb_feas[i][0];
        } else if (i % 2 == 1) {
            for (size_t j = 0; j < emb_feas[i].size(); ++j) {
                dnn_vec[j] += emb_feas[i][j];
            }
        }
    }

    int count = emb_feas.size() / 2;
    merged_feas.push_back(wide_lr / count);
    for (auto& w : dnn_vec) {
        merged_feas.push_back(w / count);
    }

    return 0;
}

int emb_lookup(const std::vector<std::vector<std::vector<uint64_t>>>& inputs,
               std::vector<std::vector<std::vector<float>>>& emb_inputs) {
    for (size_t b = 0; b < inputs.size(); ++b) {
        std::vector<std::vector<float>> emb_slots;
        for (size_t s = 0; s < inputs[b].size(); ++s) {
            int input_slot = s;
            std::vector<std::vector<float>> emb_feas;
            for (size_t f = 0; f < inputs[b][input_slot].size(); ++f) {
                std::vector<float> weight;
                // TODO : seek embedding
                InitWeight(1, weight);
                emb_feas.emplace_back(std::move(weight));
                InitWeight(8, weight);
                emb_feas.emplace_back(std::move(weight));
            }
            std::vector<float> merged_feas;
            combine_fea(emb_feas, merged_feas);
            emb_slots.emplace_back(std::move(merged_feas));
        }
        emb_inputs.emplace_back(std::move(emb_slots));
    }

    return 0;
}

int main() {
    std::string train_slot = "./data/slot.data";
    std::ifstream slot_if(train_slot);
    if (!slot_if.is_open()) {
        return -1;
    }
    std::string input;
    getline(slot_if, input);
    std::vector<std::string> slots_vec;
    boost::split(slots_vec, input, boost::is_any_of(","));
    std::map<int, int> slot2pos;
    for (size_t i = 0; i < slots_vec.size(); ++i) {
        slot2pos[std::stoi(slots_vec[i])] = i;
    }

    void* handle = dlopen("./libmodel.so", RTLD_LAZY);
    if (handle == NULL) {
        std::cerr << "dlopen error." << std::endl;
        return -1;
    }
    RUN_FUNC run_func = reinterpret_cast<RUN_FUNC>(dlsym(handle, "Run"));
    if (run_func == NULL) {
        std::cerr << "get Run error." << std::endl;
        return -1;
    }

    std::string file_name = "./data/feature.data";
    std::ifstream data_if(file_name);
    if (!data_if.is_open()) {
        std::cerr << "open input error." << std::endl;
        return -1;
    }

    std::vector<std::vector<std::vector<uint64_t>>> inputs;
    std::vector<std::vector<std::vector<float>>> emb_inputs;
    while (getline(data_if, input)) {
        std::vector<std::vector<uint64_t>> one_input;
        one_input.assign(slot2pos.size(), {});
        std::vector<std::string> vec;
        boost::split(vec, input, boost::is_any_of("\t"));
        for (size_t i = 1; i < vec.size(); ++i) {
            std::vector<uint64_t> feas;
            std::vector<std::string> slot_feas;
            boost::split(slot_feas, vec[i], boost::is_any_of("\001"));
            int index;
            // std::cout << "slot fea:" << slot_feas[0] << std::endl;
            if (slot2pos.count(std::stoi(slot_feas[0])) > 0) {
                index = slot2pos[std::stoi(slot_feas[0])];
                std::vector<std::string> feas_vec;
                boost::split(feas_vec, slot_feas[1], boost::is_any_of("\002"));
                for (auto& fea : feas_vec) {
                    feas.push_back(std::stoull(fea));
                }
            } else {
                continue;
            }
            one_input[index] = feas;
        }
        inputs.emplace_back(one_input);

        if (inputs.size() % k_batch_size == 0) {
            emb_lookup(inputs, emb_inputs);
            std::vector<float> outputs;
            auto start = system_clock::now();
            run_func(emb_inputs, outputs);
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);

            for (size_t i = 0; i < outputs.size(); ++i) {
                std::cout << outputs[i] << std::endl;
            }
            inputs.clear();
            outputs.clear();
            emb_inputs.clear();
        }
    }

    if (inputs.size() != 0) {
        emb_lookup(inputs, emb_inputs);
        std::vector<float> outputs;
        auto start = system_clock::now();
        run_func(emb_inputs, outputs);
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        for (auto w : outputs) {
            std::cout << w << std::endl;
        }
    }

    return 0;
}
