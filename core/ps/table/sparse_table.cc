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

#include "core/ps/table/sparse_table.h"

#include <set>
#include <string>

#include <butil/containers/flat_map.h>
#include <butil/logging.h>
#include <butil/object_pool.h>

#include "core/ps/optimizer/optimizer_kernel.h"
#include "core/ps/ps_cluster.h"

namespace tensornet {

SparseTable::SparseTable(const OptimizerBase* opt)
    : opt_(opt) {
    CHECK(opt_ != nullptr);

    op_kernel_ = opt_->CreateSparseOptKernel();
}

void SparseTable::SetHandle(uint32_t handle) {
    CHECK(handle_ == 0) << "sparse table handle has already set:" << handle_;

    handle_ = handle;
}

void SparseTable::Pull(const SparsePullRequest* req, SparsePullResponse* resp) {
    resp->set_table_handle(req->table_handle());
    resp->set_dim(req->dim());

    for (int i = 0; i < req->signs_size(); ++i) {
        SparseWeightInfo weight_info;
        uint64_t sign = req->signs(i);

        if (false == op_kernel_->GetWeight(sign, weight_info)) {
            weight_info.weight = nullptr;
            weight_info.dim = req->dim();
            weight_info.version = 0;

            CHECK(op_kernel_->NewSignWithWeight(sign, weight_info));
            CHECK(nullptr != weight_info.weight);
        }

        CHECK_EQ(req->dim(), weight_info.dim) << sign;

        VariableWeight* weight = resp->add_weight();

        weight->set_sign(sign);
        weight->set_version(weight_info.version);

        for (uint32_t j = 0; j < req->dim(); j++) {
            weight->add_w(weight_info.weight[j]);
        }
    }
}

void SparseTable::Push(const SparsePushRequest* req, SparsePushResponse* resp) {
    int dim = req->dim();

    std::vector<float> grad(dim);

    for (int i = 0; i < req->weight_size(); i++) {
        const VariableWeight& weight = req->weight(i);

        CHECK_EQ(weight.w_size(), dim);

        for (int j = 0; j < weight.w_size(); j++) {
            grad[j] = weight.w(j);
        }

        SparseGradInfo grad_info;
        grad_info.grad = grad.data();
        grad_info.dim = dim;
        grad_info.show = weight.show();
        grad_info.version = weight.version();

        op_kernel_->Apply(weight.sign(), grad_info);
    }
}

void SparseTable::Save(const std::string& filepath) const {
    butil::Timer timer(butil::Timer::STARTED);

    int shard_id = PsCluster::Instance()->Rank();

    std::string file = filepath + "/sparse_table/" + std::to_string(GetHandle())
                             + "/rank_" + std::to_string(shard_id);

    op_kernel_->Serialized(file);

    timer.stop();

    LOG(INFO) << "SparseTable save. rank:" << shard_id
              << " table_id:" << GetHandle()
              << " latency:" << timer.s_elapsed() << "s"
              << " keys_count:" << op_kernel_->KeyCount();
}

void SparseTable::Load(const std::string& filepath) const {
    butil::Timer timer(butil::Timer::STARTED);

    int shard_id = PsCluster::Instance()->Rank();

    std::string file = filepath + "/sparse_table/" + std::to_string(GetHandle())
                             + "/rank_" + std::to_string(shard_id);
    op_kernel_->DeSerialized(file);

    timer.stop();

    LOG(INFO) << "SparseTable load. rank:" << shard_id
              << " table_id:" << GetHandle()
              << " latency:" << timer.s_elapsed() << "s"
              << " keys_count:" << op_kernel_->KeyCount();
}

SparseTableRegistry* SparseTableRegistry::Instance() {
    static SparseTableRegistry instance;
    return &instance;
}

SparseTable* SparseTableRegistry::Get(uint32_t table_handle) {
    CHECK(table_handle < tables_.size())
        << " table_handle:" << table_handle << " table size:" << tables_.size();
    return tables_[table_handle];
}

uint32_t SparseTableRegistry::Register(SparseTable* table) {
    const std::lock_guard<std::mutex> lock(mu_);

    uint32_t table_handle = tables_.size();
    tables_.emplace_back(table);
    return table_handle;
}

SparseTable* CreateSparseTable(const OptimizerBase* opt) {
    SparseTable* table = new SparseTable(opt);

    table->SetHandle(SparseTableRegistry::Instance()->Register(table));

    return table;
}

}  // namespace tensornet
