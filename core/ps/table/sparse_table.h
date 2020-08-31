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

#ifndef TENSORNET_PS_TABLE_SPARSE_TABLE_H_
#define TENSORNET_PS_TABLE_SPARSE_TABLE_H_

#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <mutex>

#include <butil/iobuf.h>

#include "core/ps/optimizer/optimizer.h"
#include "core/protobuf/ps_server.pb.h"

namespace tensornet {

class SparseTable {
public:
    SparseTable(const OptimizerBase* opt);

    ~SparseTable() {}

    void Pull(const SparsePullRequest* req, SparsePullResponse* resp);

    void Push(const SparsePushRequest* req, SparsePushResponse* resp);

    void SetHandle(uint32_t handle);

    uint32_t GetHandle() const {
        return handle_;
    }

    size_t VariableCount() const {
        return vars_.size();
    }

    void Save(const std::string& filepath) const;

    void Load(const std::string& filepath) const;

    void ShowDecay() const;

private:
    uint32_t handle_ = 0;
    const OptimizerBase* opt_ = nullptr;
    std::shared_ptr<SparseOptimizerKernelBase> op_kernel_;
    std::set<std::string> vars_;
};

class SparseTableRegistry {
public:
    static SparseTableRegistry* Instance();

    SparseTable* Get(uint32_t table_handle);

    uint32_t Register(SparseTable* table);

private:
    SparseTableRegistry() {}
    ~SparseTableRegistry() {}

private:
    std::mutex mu_;
    std::vector<SparseTable*> tables_;
};

SparseTable* CreateSparseTable(const OptimizerBase* opt);

}  // namespace tensornet

#endif  // TENSORNET_PS_TABLE_SPARSE_TABLE_H_
