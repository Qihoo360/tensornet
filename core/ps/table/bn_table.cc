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
#include <cstdio>

#include <boost/iostreams/stream.hpp>
#include "core/ps_interface/ps_raw_interface.h"
#include "core/utility/file_io.h"
#include "core/ps/optimizer/data_struct.h"

namespace tensornet {

BnTable::BnTable(const std::string& name, int shard_num, int self_shard_id, int bn_size, bool synchronized, float moment, int max_count)
    : shard_num_(shard_num)
    , self_shard_id_(self_shard_id)
    , name_(name)
    , synchronized_(synchronized)
    , moment_(moment)
    , max_count_(max_count)
	, bn_size_(bn_size) {
    total_sum_.setZero(bn_size);
    total_squared_sum_.setZero(bn_size);
    total_count_.setZero(bn_size);
    inc_sum_.setZero(bn_size);
    inc_squared_sum_.setZero(bn_size);
    inc_count_.setZero(bn_size);
    mu_ = std::make_unique<std::mutex>();
}

void BnTable::SetHandle(uint32_t handle) {
    CHECK(handle_ == 0) << "bn table handle has already set:" << handle_;

    handle_ = handle;
}

void BnTable::Append(butil::IOBuf& bn_statistics_buf, bool isLocal) {
    // const std::lock_guard<std::mutex> lock(*mu_);
    Eigen::ArrayXf acc_sum = Eigen::ArrayXf::Zero(bn_size_); 
    Eigen::ArrayXf acc_squared_sum = Eigen::ArrayXf::Zero(bn_size_); 
    Eigen::ArrayXf acc_count = Eigen::ArrayXf::Zero(bn_size_); 
 
    bn_statistics_buf.cutn(acc_sum.data(), acc_sum.size() * sizeof(float));
    bn_statistics_buf.cutn(acc_squared_sum.data(), acc_squared_sum.size() * sizeof(float));
    bn_statistics_buf.cutn(acc_count.data(), acc_count.size() * sizeof(float));

    if(synchronized_ && isLocal){
        inc_sum_ += acc_sum;
        inc_squared_sum_ += acc_squared_sum;
        inc_count_ += acc_count;
    }

    int cur_count = static_cast<int>(total_count_.maxCoeff());
    if(cur_count > max_count_) {
        int acc_count_num = static_cast<int>(acc_count.maxCoeff());
        float ratio = (float) acc_count_num / cur_count;
        total_sum_ = total_sum_ * (1 - (1 - moment_) * ratio) +  (1 - moment_) * ratio * acc_sum;
        total_squared_sum_ = total_squared_sum_ * (1 - (1 - moment_) * ratio) + (1 - moment_) * ratio * acc_squared_sum;

    } else { 

    total_sum_ += acc_sum;
    total_squared_sum_ += acc_squared_sum;
    total_count_ += acc_count;
    }
}


std::tuple<Eigen::ArrayXf,Eigen::ArrayXf> BnTable::GetMoments() {
    Eigen::ArrayXf global_mean = DivideNoNan(total_sum_, total_count_);
    Eigen::ArrayXf global_squared_mean = DivideNoNan(total_squared_sum_, total_count_);
    Eigen::ArrayXf global_var = (global_squared_mean - global_mean.square()).max(0.0);
/*
    if(handle_ == 0) {
    std::cout << "current mean and var \n";
       for (int i = 0; i < global_mean.size(); ++i) {
        std::cout << global_mean(i) << " ";
   }
   std::cout << std::endl;
       for (int i = 0; i < global_var.size(); ++i) {
        std::cout << global_var(i) << " ";
   }
   std::cout << std::endl;
    }
*/
    return std::make_tuple(global_mean, global_var);
}

void BnTable::GetStatistics(const BnStatisticsPullRequest* req, butil::IOBuf& bn_statistics_buf, BnStatisticsPullResponse* resp) {
    resp->set_table_handle(req->table_handle());
    bn_statistics_buf.append(total_sum_.data(), total_sum_.size() * sizeof(float));
    bn_statistics_buf.append(total_squared_sum_.data(), total_squared_sum_.size() * sizeof(float));
    bn_statistics_buf.append(total_count_.data(), total_count_.size() * sizeof(float));
}

void BnTable::GetIncStatistics(butil::IOBuf& bn_statistics_buf) {
    bn_statistics_buf.append(inc_sum_.data(), inc_sum_.size() * sizeof(float));
    bn_statistics_buf.append(inc_squared_sum_.data(), inc_squared_sum_.size() * sizeof(float));
    bn_statistics_buf.append(inc_count_.data(), inc_count_.size() * sizeof(float));
	inc_sum_.setZero();
	inc_squared_sum_.setZero();
	inc_count_.setZero();
}


void BnTable::Refresh() {
        total_sum_.setZero();
        total_squared_sum_.setZero();
        total_count_.setZero();

        inc_sum_.setZero();
        inc_squared_sum_.setZero();
        inc_count_.setZero();
}


Eigen::ArrayXf BnTable::DivideNoNan(const Eigen::ArrayXf& numerator, const Eigen::ArrayXf& denominator) {  
   Eigen::ArrayXf result = numerator;
   for (int i = 0; i < numerator.size(); ++i) {  
        if (!std::isnan(denominator(i)) && denominator(i) != 0.0) {  
            result(i) = numerator(i) / denominator(i);  
        } else {  
            result(i) = 0.0;  
        }  
    }  
    return result;  
}

void BnTable::PrintDetail(){
   std::cout << "Array elements for handle: " << handle_ << " Elements: ";
   for (int i = 0; i < total_squared_sum_.size(); ++i) {
        std::cout << total_squared_sum_(i) << " ";
   }
   std::cout << std::endl;
}

void BnTable::Load(const std::string& filepath) {

    std::cout << "start to load from bn ***********************";
    std::string file = filepath + "/bn_table/";
    file += std::to_string(GetHandle());

    FileReaderSource reader_source(file, FCT_ZLIB);
    boost::iostreams::stream<FileReaderSource> in_stream(reader_source);
    in_stream.iword(SERIALIZE_FMT_ID) = SF_BIN;

    int bn_size = 0;
    in_stream.read(reinterpret_cast<char*>(&bn_size), sizeof(bn_size));
    
    for( int i = 0; i < bn_size; i++) {
        in_stream.read(reinterpret_cast<char*>(&total_sum_[i]), sizeof(total_sum_[i]));
        in_stream.read(reinterpret_cast<char*>(&total_squared_sum_[i]), sizeof(total_squared_sum_[i]));
        in_stream.read(reinterpret_cast<char*>(&total_count_[i]), sizeof(total_count_[i]));
    }

    PrintDetail();

}

void BnTable::Save(const std::string& filepath) {

    std::string file = filepath + "/bn_table/";

    file += std::to_string(GetHandle());

    FileWriterSink writer_sink(file, FCT_ZLIB);

    if(handle_ == 0) {
    std::cout << "current mean and var \n";
       for (int i = 0; i < total_sum_.size(); ++i) {
        std::cout << total_sum_(i) << " ";
   }
   std::cout << std::endl;
       for (int i = 0; i < total_squared_sum_.size(); ++i) {
        std::cout << total_squared_sum_(i) << " ";
   }
   std::cout << std::endl;
    }

    boost::iostreams::stream<FileWriterSink> out_stream(writer_sink);
    out_stream.iword(SERIALIZE_FMT_ID) = SF_BIN;

    out_stream.write(reinterpret_cast<const char*>(&bn_size_), sizeof(bn_size_));

    for( int i = 0; i < bn_size_; i++) {
        out_stream.write(reinterpret_cast<const char*>(&total_sum_[i]), sizeof(total_sum_[i]));
        out_stream.write(reinterpret_cast<const char*>(&total_squared_sum_[i]), sizeof(total_squared_sum_[i]));
        out_stream.write(reinterpret_cast<const char*>(&total_count_[i]), sizeof(total_count_[i]));
    }
    out_stream.flush();

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

BnTable* CreateBnTable(const std::string& name, int shard_num, int self_shard_id, int bn_size, bool sync, float moment, int max_count) {
    BnTable* table = new BnTable(name, shard_num, self_shard_id, bn_size, sync, moment, max_count);

    table->SetHandle(BnTableRegistry::Instance()->Register(table));

    return table;
}

}  // namespace tensornet
