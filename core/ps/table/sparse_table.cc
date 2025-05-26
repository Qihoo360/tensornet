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

#include "core/ps/table/sparse_table.h"

#include <set>
#include <string>

#include <butil/containers/flat_map.h>
#include <butil/logging.h>
#include <butil/object_pool.h>

#include "core/ps/optimizer/optimizer_kernel.h"
#include "core/ps_interface/ps_raw_interface.h"

namespace tensornet {

SparseTable::SparseTable(
    const OptimizerBase* opt, const std::string& name, int dimension, int shard_num, int self_shard_id)
    : shard_num_(shard_num)
    , self_shard_id_(self_shard_id)
    , opt_(opt)
    , dim_(dimension)
    , name_(name) {
    CHECK(opt_ != nullptr);

    op_kernel_ = opt_->CreateSparseOptKernel(dim_);
}

void SparseTable::SetHandle(uint32_t handle) {
    CHECK(handle_ == 0) << "sparse table handle has already set:" << handle_;

    handle_ = handle;
}

void SparseTable::Pull(const SparsePullRequest* req, butil::IOBuf& out_emb_buf, SparsePullResponse* resp) {
    resp->set_table_handle(req->table_handle());

    resp->set_dim(req->dim());

    for (int i = 0; i < req->signs_size(); ++i) {
        uint64_t sign = req->signs(i);

        // w.size() is guaranteed by op_kernel_ same with dim_
        float* w = op_kernel_->GetWeight(sign);
        CHECK(nullptr != w);

        out_emb_buf.append(w, sizeof(float) * (req->dim()));
    }
}

void SparseTable::Push(const SparsePushRequest* req, butil::IOBuf& grad_buf, SparsePushResponse* resp) {
    float grad[req->dim()];
    SparsePushSignInfo sign_info;

    while (sizeof(sign_info) == grad_buf.cutn(&sign_info, sizeof(sign_info))) {
        size_t grad_size = sizeof(float) * req->dim();
        CHECK_EQ(grad_size, grad_buf.cutn(grad, grad_size));

        SparseGradInfo grad_info;
        grad_info.grad = grad;
        grad_info.batch_show = sign_info.batch_show;
        grad_info.batch_click = sign_info.batch_click;

        op_kernel_->Apply(sign_info.sign, grad_info);
    }
}

void SparseTable::Save(const std::string& filepath, const std::string& mode) {
    butil::Timer timer(butil::Timer::STARTED);

    std::string file = filepath + "/sparse_table/";

    if (name_.empty()) {
        file += std::to_string(GetHandle());
    } else {
        file += name_;
    }

    file += "/rank_" + std::to_string(self_shard_id_);

    op_kernel_->Serialized(file, mode);

    timer.stop();

    int new_key_count = op_kernel_->KeyCount();

    LOG(INFO) << "SparseTable save. rank:" << self_shard_id_ << " name:" << name_ << " handle:" << GetHandle()
              << " latency:" << timer.s_elapsed() << "s"
              << " key_count:" << new_key_count << " increased key_count:" << new_key_count - saved_key_count_;

    saved_key_count_ = new_key_count;
}

void SparseTable::Load(const std::string& filepath, const std::string& mode) {
    butil::Timer timer(butil::Timer::STARTED);

    std::string file = filepath + "/sparse_table/";

    if (name_.empty()) {
        file += std::to_string(GetHandle());
    } else {
        if (FileUtils::CheckFileExists(file + name_)) {
            file += name_;
        } else {
            file += std::to_string(GetHandle());
        }
    }

    file += "/rank_" + std::to_string(self_shard_id_);

    op_kernel_->DeSerialized(file, mode);

    timer.stop();

    saved_key_count_ = op_kernel_->KeyCount();

    LOG(INFO) << "SparseTable load. rank:" << self_shard_id_ << " name:" << name_ << " handle:" << GetHandle()
              << " latency:" << timer.s_elapsed() << "s"
              << " key_count:" << saved_key_count_;
}

void SparseTable::ShowDecay(int delta_days) const { op_kernel_->ShowDecay(delta_days); }

SparseTableRegistry* SparseTableRegistry::Instance() {
    static SparseTableRegistry instance;
    return &instance;
}

SparseTable* SparseTableRegistry::Get(uint32_t table_handle) {
    CHECK(table_handle < tables_.size()) << " table_handle:" << table_handle << " table size:" << tables_.size();
    return tables_[table_handle];
}

uint32_t SparseTableRegistry::Register(SparseTable* table) {
    const std::lock_guard<std::mutex> lock(mu_);

    uint32_t table_handle = tables_.size();
    tables_.emplace_back(table);
    return table_handle;
}

SparseTable* CreateSparseTable(
    const OptimizerBase* opt, const std::string& name, int dimension, int shard_num, int self_shard_id) {
    SparseTable* table = new SparseTable(opt, name, dimension, shard_num, self_shard_id);

    table->SetHandle(SparseTableRegistry::Instance()->Register(table));

    return table;
}

}  // namespace tensornet
