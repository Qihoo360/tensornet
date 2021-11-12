// Copyright (c) 2020, Qihoo, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/ps/ps_cluster.h"

#include "core/ps/optimizer/optimizer.h"
#include "core/ps/table/dense_table.h"
#include "core/ps/table/sparse_table.h"
#include "core/kernels/data/balance_dataset_ops.h"

#include <memory>

#include <Python.h>

#include <pybind11/pybind11.h>
#include <butil/logging.h>

namespace py = pybind11;

using std::vector;
using std::string;

using namespace tensornet;

#define PYDICT_PARSE_KWARGS(kwargs, name, default_value)                        \
    opt->name = default_value;                                                  \
    {                                                                           \
        PyObject* item = PyDict_GetItemString(kwargs.ptr(), #name);             \
        if (NULL != item) {                                                     \
            opt->name = PyFloat_AsDouble(item);                                 \
        }                                                                       \
    }

PYBIND11_MODULE(_pywrap_tn, m) {
    m.def("init", []() {
        PsCluster* cluster = PsCluster::Instance();

        if (cluster->IsInitialized()) {
            return true;
        }

        if (cluster->Init() < 0) {
            throw py::value_error("Init tensornet fail");
        }

        tensorflow::BalanceInputDataInfo* data_info = tensorflow::BalanceInputDataInfo::Instance();

        if (data_info->Init(cluster->RankNum(), cluster->Rank()) < 0) {
            throw py::value_error("Init BalanceInputDataInfo fail");
        }

        return true;
    })
    .def("AdaGrad", [](py::kwargs kwargs) {
        auto opt = new AdaGrad();

        PYDICT_PARSE_KWARGS(kwargs, learning_rate, 0.01);
        PYDICT_PARSE_KWARGS(kwargs, show_decay_rate, 0.98);
        PYDICT_PARSE_KWARGS(kwargs, feature_drop_show, 1 - opt->show_decay_rate);

        PYDICT_PARSE_KWARGS(kwargs, initial_g2sum, 0);
        PYDICT_PARSE_KWARGS(kwargs, initial_scale, 1);
        PYDICT_PARSE_KWARGS(kwargs, epsilon, 1e-8);
        PYDICT_PARSE_KWARGS(kwargs, grad_decay_rate, 1.0);
        PYDICT_PARSE_KWARGS(kwargs, mom_decay_rate, 1.0);
        PYDICT_PARSE_KWARGS(kwargs, no_show_days, 1000);

        // NOTICE! opt will not delete until system exist
        PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

        return py::reinterpret_steal<py::object>(obj);
    })
    .def("Adam", [](py::kwargs kwargs) {
        auto opt = new Adam();

        PYDICT_PARSE_KWARGS(kwargs, learning_rate, 0.001);
        PYDICT_PARSE_KWARGS(kwargs, show_decay_rate, 0.98);
        PYDICT_PARSE_KWARGS(kwargs, feature_drop_show, 1 - opt->show_decay_rate);

        PYDICT_PARSE_KWARGS(kwargs, beta1, 0.9);
        PYDICT_PARSE_KWARGS(kwargs, beta2, 0.999);
        PYDICT_PARSE_KWARGS(kwargs, epsilon, 1e-8);
        PYDICT_PARSE_KWARGS(kwargs, initial_scale, 1.0);

        // NOTICE! opt will not delete until system exist
        PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

        return py::reinterpret_steal<py::object>(obj);
    })
    .def("Ftrl", [](py::kwargs kwargs) {
        auto opt = new Ftrl();
        PYDICT_PARSE_KWARGS(kwargs, learning_rate, 0.05);
        PYDICT_PARSE_KWARGS(kwargs, show_decay_rate, 0.98);
        PYDICT_PARSE_KWARGS(kwargs, show_threshold, 0.0);
        PYDICT_PARSE_KWARGS(kwargs, feature_drop_show, 1 - opt->show_decay_rate);

        PYDICT_PARSE_KWARGS(kwargs, beta, 1);
        PYDICT_PARSE_KWARGS(kwargs, lambda1, 0.1);
        PYDICT_PARSE_KWARGS(kwargs, lambda2, 1);
        PYDICT_PARSE_KWARGS(kwargs, initial_scale, 1.0);

        // NOTICE! opt will not delete until system exist
        PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

        return py::reinterpret_steal<py::object>(obj);
    })
    .def("create_sparse_table", [](py::object obj, std::string name, int dimension) {
        OptimizerBase* opt =
               static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));

        PsCluster* cluster = PsCluster::Instance();

        SparseTable* table = CreateSparseTable(opt, name, dimension, cluster->RankNum(), cluster->Rank());

        return table->GetHandle();
    })
    .def("create_dense_table", [](py::object obj) {
         OptimizerBase* opt =
               static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));

        PsCluster* cluster = PsCluster::Instance();

        DenseTable* table = CreateDenseTable(opt, cluster->RankNum(), cluster->Rank());

        return table->GetHandle();
    })
    .def("save_sparse_table", [](uint32_t table_handle, std::string filepath,
                const std::string& mode="txt") {
        SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
        return table->Save(filepath, mode);
    })
    .def("load_sparse_table", [](uint32_t table_handle, std::string filepath,
                const std::string& mode="txt") {
        SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
        return table->Load(filepath, mode);
    })
    .def("save_dense_table", [](uint32_t table_handle, std::string filepath) {
        DenseTable* table = DenseTableRegistry::Instance()->Get(table_handle);
        return table->Save(filepath);
    })
    .def("load_dense_table", [](uint32_t table_handle, std::string filepath) {
        DenseTable* table = DenseTableRegistry::Instance()->Get(table_handle);
        return table->Load(filepath);
    })
    .def("shard_num", []() {
        PsCluster* cluster = PsCluster::Instance();

        return cluster->RankNum();
    })
    .def("self_shard_id", []() {
        PsCluster* cluster = PsCluster::Instance();

        return cluster->Rank();
    })
    .def("barrier", []() {
        PsCluster* cluster = PsCluster::Instance();

        cluster->Barrier();

        return;
    })
    .def("reset_balance_dataset", []() {
        PsCluster* cluster = PsCluster::Instance();
        tensorflow::BalanceInputDataInfo* data_info = tensorflow::BalanceInputDataInfo::Instance();
        if (data_info->Init(cluster->RankNum(), cluster->Rank()) < 0) {
            throw py::value_error("reset_balance_dataset fail");
        }
    })
    .def("show_decay", [](uint32_t table_handle, int delta_days) {
        SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
        return table->ShowDecay(delta_days);
    })
    ;
};
