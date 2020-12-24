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
        float learning_rate = 0.01;
        float initial_g2sum = 0;
        float initial_scale = 1;
        float epsilon = 1e-8;
        float grad_decay_rate = 1.0;
        float mom_decay_rate = 1.0;
        float show_decay_rate = 0.98;

        PyObject* item = PyDict_GetItemString(kwargs.ptr(), "learning_rate");
        if (NULL != item) {
            learning_rate = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "initial_g2sum");
        if (NULL != item) {
            initial_g2sum = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "initial_scale");
        if (NULL != item) {
            initial_scale = PyFloat_AsDouble(item);
        }
        item = PyDict_GetItemString(kwargs.ptr(), "epsilon");
        if (NULL != item) {
            epsilon = PyFloat_AsDouble(item);
        }
        item = PyDict_GetItemString(kwargs.ptr(), "grad_decay_rate");
        if (NULL != item) {
            grad_decay_rate = PyFloat_AsDouble(item);
        }
        item = PyDict_GetItemString(kwargs.ptr(), "mom_decay_rate");
        if (NULL != item) {
            mom_decay_rate = PyFloat_AsDouble(item);
        }
        item = PyDict_GetItemString(kwargs.ptr(), "show_decay_rate");
        if (NULL != item) {
            show_decay_rate = PyFloat_AsDouble(item);
        }

        auto opt = new AdaGrad(learning_rate, initial_g2sum, initial_scale, epsilon, 
                grad_decay_rate, mom_decay_rate, show_decay_rate);

        // NOTICE! opt will not delete until system exist
        PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

        return py::reinterpret_steal<py::object>(obj);
    })
    .def("Adam", [](py::kwargs kwargs) {
        float learning_rate = 0.001;
        float beta1 = 0.9;
        float beta2 = 0.999;
        float epsilon = 1e-8;
        float initial_scale = 1.0;

        PyObject* item = PyDict_GetItemString(kwargs.ptr(), "learning_rate");
        if (NULL != item) {
            learning_rate = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "beta1");
        if (NULL != item) {
            beta1 = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "beta2");
        if (NULL != item) {
            beta2 = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "epsilon");
        if (NULL != item) {
            epsilon = PyFloat_AsDouble(item);
        }
        item = PyDict_GetItemString(kwargs.ptr(), "initial_scale");
        if (NULL != item) {
            initial_scale = PyFloat_AsDouble(item);
        }

        auto opt = new Adam(learning_rate, beta1, beta2, epsilon, initial_scale);

        // NOTICE! opt will not delete until system exist
        PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

        return py::reinterpret_steal<py::object>(obj);
    })
    .def("Ftrl", [](py::kwargs kwargs) {
        float learning_rate = 0.05;
        float initial_range = 0;
        float beta = 1;
        float lambda1 = 0.1;
        float lambda2 = 1;
        float show_decay_rate = 0.98;

        PyObject* item = PyDict_GetItemString(kwargs.ptr(), "learning_rate");
        if (NULL != item) {
            learning_rate = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "initial_range");
        if (NULL != item) {
            initial_range = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "beta");
        if (NULL != item) {
            beta = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "lambda1");
        if (NULL != item) {
            lambda1 = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "lambda2");
        if (NULL != item) {
            lambda2 = PyFloat_AsDouble(item);
        }

        item = PyDict_GetItemString(kwargs.ptr(), "show_decay_rate");
        if (NULL != item) {
            show_decay_rate = PyFloat_AsDouble(item);
        }

        auto opt = new Ftrl(learning_rate, initial_range, beta, lambda1, lambda2, show_decay_rate);

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
    .def("save_sparse_table", [](uint32_t table_handle, std::string filepath) {
        SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
        return table->Save(filepath);
    })
    .def("load_sparse_table", [](uint32_t table_handle, std::string filepath) {
        SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
        return table->Load(filepath);
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
    .def("show_decay", [](uint32_t table_handle) {
        SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
        return table->ShowDecay();
    })
    ;
};
