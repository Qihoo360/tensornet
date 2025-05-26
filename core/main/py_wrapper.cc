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

#include "core/utility/fix_redef.h"

#include "core/ps/ps_cluster.h"

#include "core/kernels/data/balance_dataset_ops.h"
#include "core/ps/optimizer/optimizer.h"
#include "core/ps/table/bn_table.h"
#include "core/ps/table/dense_table.h"
#include "core/ps/table/sparse_table.h"

#include <memory>

#include <Python.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

using std::string;
using std::vector;

using namespace tensornet;

#define PYDICT_PARSE_KWARGS(kwargs, name, default_value)            \
    opt->name = default_value;                                      \
    {                                                               \
        PyObject* item = PyDict_GetItemString(kwargs.ptr(), #name); \
        if (NULL != item) {                                         \
            opt->name = PyFloat_AsDouble(item);                     \
        }                                                           \
    }

#define PYDICT_PARSE_LEARNING_RATE(kwargs)                                                      \
    do {                                                                                        \
        PyObject* lr_item = PyDict_GetItemString((kwargs).ptr(), "learning_rate");              \
        if (lr_item) {                                                                          \
            if (PyFloat_Check(lr_item)) {                                                       \
                opt->SetSchedule(PyFloat_AsDouble(lr_item));                                    \
            } else {                                                                            \
                py::object schedule = py::reinterpret_borrow<py::object>(lr_item);              \
                opt->SetSchedule(std::unique_ptr<py::object, void (*)(py::object*)>(            \
                    new py::object(std::move(schedule)), [](py::object* obj) { delete obj; })); \
            }                                                                                   \
        }                                                                                       \
    } while (0)

PYBIND11_MODULE(_pywrap_tn, m) {
    m.def("init",
          []() {
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
        .def("AdaGrad",
             [](py::kwargs kwargs) {
                 auto opt = new AdaGrad();

                 PYDICT_PARSE_LEARNING_RATE(kwargs);
                 PYDICT_PARSE_KWARGS(kwargs, show_decay_rate, 0.98);
                 PYDICT_PARSE_KWARGS(kwargs, show_threshold, 0.0);

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
        .def("Adam",
             [](py::kwargs kwargs) {
                 auto opt = new Adam();

                 PYDICT_PARSE_LEARNING_RATE(kwargs);
                 PYDICT_PARSE_KWARGS(kwargs, show_decay_rate, 0.98);

                 PYDICT_PARSE_KWARGS(kwargs, beta1, 0.9);
                 PYDICT_PARSE_KWARGS(kwargs, beta2, 0.999);
                 PYDICT_PARSE_KWARGS(kwargs, epsilon, 1e-8);
                 PYDICT_PARSE_KWARGS(kwargs, initial_scale, 1.0);

                 // NOTICE! opt will not delete until system exist
                 PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

                 return py::reinterpret_steal<py::object>(obj);
             })
        .def("get_opt_learning_rate",
             [](py::object obj) {
                 OptimizerBase* opt = static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));

                 auto scheduler_ptr = opt->GetSchedule();

                 if (scheduler_ptr) {
                     return *scheduler_ptr;
                 } else {
                     return py::cast(opt->learning_rate);
                 }
             })
        .def("Ftrl",
             [](py::kwargs kwargs) {
                 auto opt = new Ftrl();
                 PYDICT_PARSE_LEARNING_RATE(kwargs);
                 PYDICT_PARSE_KWARGS(kwargs, show_decay_rate, 0.98);
                 PYDICT_PARSE_KWARGS(kwargs, show_threshold, 0.0);

                 PYDICT_PARSE_KWARGS(kwargs, beta, 1);
                 PYDICT_PARSE_KWARGS(kwargs, lambda1, 0.1);
                 PYDICT_PARSE_KWARGS(kwargs, lambda2, 1);
                 PYDICT_PARSE_KWARGS(kwargs, initial_scale, 1.0);

                 // NOTICE! opt will not delete until system exist
                 PyObject* obj = PyCapsule_New(opt, nullptr, nullptr);

                 return py::reinterpret_steal<py::object>(obj);
             })
        .def("set_sparse_init_mode",
             [](py::object obj, bool is_training) -> void {
                 OptimizerBase* opt = static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));

                 opt->SetSparseZeroInit(!is_training);
             })
        .def("create_sparse_table",
             [](py::object obj, std::string name, int dimension, bool use_cvm) {
                 OptimizerBase* opt = static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));

                 opt->SetUseCvm(use_cvm);

                 PsCluster* cluster = PsCluster::Instance();

                 SparseTable* table = CreateSparseTable(opt, name, dimension, cluster->RankNum(), cluster->Rank());

                 return table->GetHandle();
             })
        .def("create_dense_table",
             [](py::object obj) {
                 OptimizerBase* opt = static_cast<OptimizerBase*>(PyCapsule_GetPointer(obj.ptr(), nullptr));

                 PsCluster* cluster = PsCluster::Instance();

                 DenseTable* table = CreateDenseTable(opt, cluster->RankNum(), cluster->Rank());

                 return table->GetHandle();
             })
        .def("create_bn_table",
             [](std::string name, uint32_t bn_size, bool sync, float moment, uint64_t max_count, bool use_pctr_dnn_bn) {
                 PsCluster* cluster = PsCluster::Instance();

                 BnTable* table = CreateBnTable(name, cluster->RankNum(), cluster->Rank(), bn_size, sync, moment,
                                                max_count, use_pctr_dnn_bn);

                 return table->GetHandle();
             })
        .def("save_bn_table",
             [](uint32_t table_handle, std::string filepath) {
                 BnTable* table = BnTableRegistry::Instance()->Get(table_handle);
                 return table->Save(filepath);
             })
        .def("load_bn_table",
             [](uint32_t table_handle, std::string filepath) {
                 BnTable* table = BnTableRegistry::Instance()->Get(table_handle);
                 return table->Load(filepath);
             })
        .def("save_sparse_table",
             [](uint32_t table_handle, std::string filepath, const std::string& mode = "txt") {
                 SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
                 return table->Save(filepath, mode);
             })
        .def("load_sparse_table",
             [](uint32_t table_handle, std::string filepath, const std::string& mode = "txt") {
                 SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
                 return table->Load(filepath, mode);
             })
        .def("save_dense_table",
             [](uint32_t table_handle, std::string filepath) {
                 DenseTable* table = DenseTableRegistry::Instance()->Get(table_handle);
                 return table->Save(filepath);
             })
        .def("load_dense_table",
             [](uint32_t table_handle, std::string filepath) {
                 DenseTable* table = DenseTableRegistry::Instance()->Get(table_handle);
                 return table->Load(filepath);
             })
        .def("shard_num",
             []() {
                 PsCluster* cluster = PsCluster::Instance();

                 return cluster->RankNum();
             })
        .def("self_shard_id",
             []() {
                 PsCluster* cluster = PsCluster::Instance();

                 return cluster->Rank();
             })
        .def("barrier",
             []() {
                 PsCluster* cluster = PsCluster::Instance();

                 cluster->Barrier();

                 return;
             })
        .def("reset_balance_dataset",
             []() {
                 PsCluster* cluster = PsCluster::Instance();
                 tensorflow::BalanceInputDataInfo* data_info = tensorflow::BalanceInputDataInfo::Instance();
                 if (data_info->Init(cluster->RankNum(), cluster->Rank()) < 0) {
                     throw py::value_error("reset_balance_dataset fail");
                 }
             })
        .def("show_decay", [](uint32_t table_handle, int delta_days) {
            SparseTable* table = SparseTableRegistry::Instance()->Get(table_handle);
            return table->ShowDecay(delta_days);
        });
};
