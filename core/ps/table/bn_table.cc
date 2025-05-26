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

#include "core/ps/table/bn_table.h"

#include <set>
#include <string>

#include <butil/containers/flat_map.h>
#include <butil/logging.h>
#include <butil/object_pool.h>
#include <cstdio>

#include <boost/iostreams/stream.hpp>
#include "core/ps/optimizer/data_struct.h"
#include "core/ps_interface/ps_raw_interface.h"
#include "core/utility/file_io.h"

namespace tensornet {

BnTable::BnTable(const std::string& name,
                 int shard_num,
                 int self_shard_id,
                 int bn_size,
                 bool synchronized,
                 float moment,
                 uint64_t max_count,
                 bool use_pctr_dnn_bn)
    : shard_num_(shard_num)
    , self_shard_id_(self_shard_id)
    , name_(name)
    , synchronized_(synchronized)
    , moment_(moment)
    , max_count_(max_count)
    , bn_size_(bn_size)
    , use_pctr_dnn_bn_(use_pctr_dnn_bn) {
    total_sum_.setZero(bn_size);
    total_sum_err_.setZero(bn_size);
    total_squared_sum_.setZero(bn_size);
    total_squared_sum_err_.setZero(bn_size);
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
    const std::lock_guard<std::mutex> lock(*mu_);
    Eigen::ArrayXd acc_sum = Eigen::ArrayXd::Zero(bn_size_);
    Eigen::ArrayXd acc_squared_sum = Eigen::ArrayXd::Zero(bn_size_);
    Eigen::ArrayXd acc_count = Eigen::ArrayXd::Zero(bn_size_);

    bn_statistics_buf.cutn(acc_sum.data(), acc_sum.size() * sizeof(double));
    bn_statistics_buf.cutn(acc_squared_sum.data(), acc_squared_sum.size() * sizeof(double));
    bn_statistics_buf.cutn(acc_count.data(), acc_count.size() * sizeof(double));
    CHECK_EQ(bn_statistics_buf.size(), 0);

    if (isLocal) {
        inc_sum_ += acc_sum;
        inc_squared_sum_ += acc_squared_sum;
        inc_count_ += acc_count;
    }

    uint64_t cur_count = static_cast<uint64_t>(total_count_.maxCoeff());

    if (max_count_ > 0 && cur_count > max_count_) {
        uint64_t acc_count_num = static_cast<uint64_t>(acc_count.maxCoeff());
        double ratio = (double)acc_count_num / cur_count;
        total_sum_ *= (1 - (1 - moment_) * ratio);
        TotalSumAcc((1 - moment_) * ratio * acc_sum);
        total_squared_sum_ *= (1 - (1 - moment_) * ratio);
        TotalSquareSumAcc((1 - moment_) * ratio * acc_squared_sum);
    } else {
        TotalSumAcc(acc_sum);
        TotalSquareSumAcc(acc_squared_sum);
        total_count_ += acc_count;
    }
}

void BnTable::TotalSquareSumAcc(Eigen::ArrayXd acc) {
    Eigen::ArrayXd y = acc - total_squared_sum_err_;
    Eigen::ArrayXd t = total_squared_sum_ + y;
    total_squared_sum_err_ = (t - total_squared_sum_) - y;
    total_squared_sum_ = t;
}

void BnTable::TotalSumAcc(Eigen::ArrayXd acc) {
    Eigen::ArrayXd y = acc - total_sum_err_;
    Eigen::ArrayXd t = total_sum_ + y;
    total_sum_err_ = (t - total_sum_) - y;
    total_sum_ = t;
}

std::tuple<Eigen::ArrayXf, Eigen::ArrayXf> BnTable::GetMoments() {
    Eigen::ArrayXf global_mean = DivideNoNan(total_sum_, total_count_);
    if (use_pctr_dnn_bn_) {
        return std::make_tuple(global_mean, total_squared_sum_.cast<float>());
    } else {
        Eigen::ArrayXf global_squared_mean = DivideNoNan(total_squared_sum_, total_count_);
        Eigen::ArrayXf global_var = (global_squared_mean - global_mean.square()).max(0.0);
        return std::make_tuple(global_mean, global_var);
    }
}

void BnTable::GetStatistics(const BnStatisticsPullRequest* req,
                            butil::IOBuf& bn_statistics_buf,
                            BnStatisticsPullResponse* resp) {
    resp->set_table_handle(req->table_handle());
    bn_statistics_buf.append(total_sum_.data(), total_sum_.size() * sizeof(double));
    bn_statistics_buf.append(total_squared_sum_.data(), total_squared_sum_.size() * sizeof(double));
    bn_statistics_buf.append(total_count_.data(), total_count_.size() * sizeof(double));
}

void BnTable::GetIncStatistics(butil::IOBuf& bn_statistics_buf) {
    bn_statistics_buf.append(inc_sum_.data(), inc_sum_.size() * sizeof(double));
    bn_statistics_buf.append(inc_squared_sum_.data(), inc_squared_sum_.size() * sizeof(double));
    bn_statistics_buf.append(inc_count_.data(), inc_count_.size() * sizeof(double));
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

Eigen::ArrayXf BnTable::DivideNoNan(const Eigen::ArrayXd& numerator, const Eigen::ArrayXd& denominator) {
    Eigen::ArrayXd result = numerator;
    for (int i = 0; i < numerator.size(); ++i) {
        if (!std::isnan(denominator(i)) && denominator(i) != 0.0) {
            result(i) = numerator(i) / denominator(i);
        } else {
            result(i) = 0.0;
        }
    }
    return result.cast<float>();
}

void BnTable::PrintDetail() {
    std::cout << "Array elements for handle: " << handle_ << " Elements: ";
    for (int i = 0; i < total_squared_sum_.size(); ++i) {
        std::cout << total_squared_sum_(i) << " ";
    }
    std::cout << std::endl;
}

void BnTable::Load(const std::string& filepath) {
    std::string file = filepath + "/bn_table/";
    file += std::to_string(GetHandle());

    FileReaderSource reader_source(file, FCT_ZLIB);
    boost::iostreams::stream<FileReaderSource> in_stream(reader_source);
    in_stream.iword(SERIALIZE_FMT_ID) = SF_BIN;

    int bn_size = 0;
    bool use_pctr_dnn_bn = false;

    in_stream.read(reinterpret_cast<char*>(&use_pctr_dnn_bn), sizeof(use_pctr_dnn_bn));
    CHECK_EQ(use_pctr_dnn_bn_, use_pctr_dnn_bn)
        << "bn calculate logic should be same, before use pctrdnn is " << use_pctr_dnn_bn;

    in_stream.read(reinterpret_cast<char*>(&bn_size), sizeof(bn_size));

    for (int i = 0; i < bn_size; i++) {
        in_stream.read(reinterpret_cast<char*>(&total_sum_[i]), sizeof(total_sum_[i]));
        in_stream.read(reinterpret_cast<char*>(&total_squared_sum_[i]), sizeof(total_squared_sum_[i]));
        in_stream.read(reinterpret_cast<char*>(&total_count_[i]), sizeof(total_count_[i]));
    }
}

void BnTable::Save(const std::string& filepath) {
    std::string file = filepath + "/bn_table/";

    file += std::to_string(GetHandle());

    FileWriterSink writer_sink(file, FCT_ZLIB);

    boost::iostreams::stream<FileWriterSink> out_stream(writer_sink);
    out_stream.iword(SERIALIZE_FMT_ID) = SF_BIN;

    out_stream.write(reinterpret_cast<const char*>(&use_pctr_dnn_bn_), sizeof(use_pctr_dnn_bn_));
    out_stream.write(reinterpret_cast<const char*>(&bn_size_), sizeof(bn_size_));

    for (int i = 0; i < bn_size_; i++) {
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
    CHECK(table_handle < tables_.size()) << " table_handle:" << table_handle << " table size:" << tables_.size();
    return tables_[table_handle];
}

uint32_t BnTableRegistry::Register(BnTable* table) {
    const std::lock_guard<std::mutex> lock(mu_);

    uint32_t table_handle = tables_.size();
    tables_.emplace_back(table);
    return table_handle;
}

BnTable* CreateBnTable(const std::string& name,
                       int shard_num,
                       int self_shard_id,
                       int bn_size,
                       bool sync,
                       float moment,
                       uint64_t max_count,
                       bool use_pctr_dnn_bn) {
    BnTable* table = new BnTable(name, shard_num, self_shard_id, bn_size, sync, moment, max_count, use_pctr_dnn_bn);

    table->SetHandle(BnTableRegistry::Instance()->Register(table));

    return table;
}

}  // namespace tensornet
