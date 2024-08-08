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

#include "core/ps/table/bn_table.h"

#include <set>
#include <string>

#include <butil/containers/flat_map.h>
#include <butil/logging.h>
#include <butil/object_pool.h>

#include "core/ps/optimizer/optimizer_kernel.h"
#include "core/ps_interface/ps_raw_interface.h"

namespace tensornet {

BnTable::BnTable(const std::string& name, int shard_num, int self_shard_id, int bn_size)
    : shard_num_(shard_num)
    , self_shard_id_(self_shard_id)
    , name_(name)
	, bn_size_(bn_size) {
	moving_mean_.setZero(bn_size);
	moving_var_.setZero(bn_size);
	batch_count_.setZero(1);
}

void BnTable::SetHandle(uint32_t handle) {
    CHECK(handle_ == 0) << "bn table handle has already set:" << handle_;

    handle_ = handle;
}

void BnTable::Pull(const BnVarsPullRequest* req, butil::IOBuf& out_vars_buf, BnVarsPullResponse* resp) {
    resp->set_table_handle(req->table_handle());
    resp->set_resp_shard_id(req->req_shard_id());
    
    out_vars_buf.append(moving_mean_.data(), moving_mean_.size() * sizeof(float));
    out_vars_buf.append(moving_var_.data(), moving_var_.size() * sizeof(float));
    out_vars_buf.append(batch_count_.data(), batch_count_.size() * sizeof(float));
}

int BnTable::Set(butil::IOBuf& bn_vars_buf){
	CHECK_EQ( (bn_size_ * 2 + 1) * sizeof(float), bn_vars_buf.size());

    bn_vars_buf.cutn(moving_mean_.data(), moving_mean_.size() * sizeof(float));
    bn_vars_buf.cutn(moving_var_.data(), moving_var_.size() * sizeof(float));
    bn_vars_buf.cutn(batch_count_.data(), batch_count_.size() * sizeof(float));
    return 0;
}


BnTableRegistry* BnTableRegistry::Instance() {
    static BnTableRegistry instance;
    return &instance;
}

BnTable* BnTableRegistry::Get(uint32_t table_handle) {
    CHECK(table_handle < tables_.size())
        << " table_handle:" << table_handle << " table size:" << tables_.size();
    return tables_[table_handle];
}

uint32_t BnTableRegistry::Register(BnTable* table) {
    const std::lock_guard<std::mutex> lock(mu_);

    uint32_t table_handle = tables_.size();
    tables_.emplace_back(table);
    return table_handle;
}

BnTable* CreateBnTable(const std::string& name, int shard_num, int self_shard_id, int bn_size) {
    BnTable* table = new BnTable(name, shard_num, self_shard_id, bn_size);

    table->SetHandle(BnTableRegistry::Instance()->Register(table));

    return table;
}

}  // namespace tensornet
