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

#ifndef TENSORNET_PS_TABLE_BN_TABLE_H_
#define TENSORNET_PS_TABLE_BN_TABLE_H_

#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <mutex>

#include <butil/iobuf.h>
#include <Eigen/Dense>

#include "core/ps_interface/ps_server.pb.h"

namespace tensornet {

class BnTable {
public:
    BnTable(const std::string& name,int shard_num, int self_shard_id, int bn_size);

    ~BnTable() = default;

    void Append(butil::IOBuf& out_buf);

    void GetStatistics(const BnStatisticsPullRequest* req, butil::IOBuf& out_buf, BnStatisticsPullResponse* resp);

    std::tuple<Eigen::ArrayXf,Eigen::ArrayXf> GetMoments();

    Eigen::ArrayXf DivideNoNan(const Eigen::ArrayXf& numerator, const Eigen::ArrayXf& denominator);

    void PrintDetail();

    void SetHandle(uint32_t handle);

    uint32_t GetHandle() const {
        return handle_;
    }

private:
    int shard_num_ = 0;
    int self_shard_id_ = 0;
    uint32_t handle_ = 0;
    std::string name_;
    uint32_t bn_size_ = 0;
    Eigen::ArrayXf total_sum_;
    Eigen::ArrayXf total_squared_sum_;
    Eigen::ArrayXf total_count_; 
    std::unique_ptr<std::mutex> mu_;
	
};

class BnTableRegistry {
public:
    ~BnTableRegistry() {
        for (auto table : tables_) {
            delete table;
        }
    }

    static BnTableRegistry* Instance();

    BnTable* Get(uint32_t table_handle);

    uint32_t Register(BnTable* table);

private:
    BnTableRegistry() { }

private:
    std::mutex mu_;
    std::vector<BnTable*> tables_;
};

BnTable* CreateBnTable(const std::string& name, int shard_num, int self_shard_id, int bn_size);

}  // namespace tensornet

#endif  // TENSORNET_PS_TABLE_BN_TABLE_H_
