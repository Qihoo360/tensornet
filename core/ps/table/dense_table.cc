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

#include "core/ps/table/dense_table.h"
#include "core/ps/optimizer/optimizer.h"
#include "core/ps/optimizer/optimizer_kernel.h"
#include "core/ps/ps_cluster.h"
#include "core/utility/file_io.h"

#include <cmath>
#include <functional>

#include <butil/logging.h>
#include <butil/iobuf.h>

namespace tensornet {

DenseTable::DenseTable(const OptimizerBase* opt)
    : total_elements_(0)
    , opt_(opt) {
    CHECK(opt_ != nullptr);
}

int DenseTable::Init(int total_elements) {
    if (is_initialized_) {
        CHECK(total_elements == total_elements_);
        return 0;
    }

    int shard_num = PsCluster::Instance()->RankNum();
    int every_kernel_size = std::ceil(total_elements * 1.0 / shard_num);

    for (int i = 0; i < shard_num; i++) {
        int offset_begin = i * every_kernel_size;
        int offset_end = (i + 1) * every_kernel_size;

        if (offset_begin >= total_elements) {
            continue;
        }

        if (offset_end > total_elements) {
            offset_end = total_elements;
        }

        if (i == PsCluster::Instance()->Rank()) {
            LOG(INFO) << "init dense table:" << GetHandle() << " shard_id:" << i
                      << " elements:" << total_elements
                      << " begin:" << offset_begin << " end:" << offset_end;
        }

        opt_kernels_.emplace_back(opt_->CreateDenseOptKernel(offset_begin, offset_end));
    }

    total_elements_ = total_elements;

    is_initialized_ = true;

    return 0;
}

int DenseTable::SetWeight(butil::IOBuf& w_buf) {
    for (size_t i = 0; i < opt_kernels_.size(); i++) {
        butil::IOBuf buf;
        int data_length = opt_kernels_[i]->Length() * sizeof(float);

        CHECK_EQ(data_length, w_buf.cutn(&buf, data_length));
        opt_kernels_[i]->SetWeight(buf);
    }

    CHECK_EQ(w_buf.size(), 0);

    return 0;
}

void DenseTable::SetHandle(uint32_t handle) {
    CHECK_EQ(handle_, 0) << "dense table handle has already set:" << handle_;

    handle_ = handle;
}

const DenseOptKernelSharedPtr DenseTable::GetOptKernels(int shard_id) const {
    if (shard_id >= (int)opt_kernels_.size()) {
        return nullptr;
    } else {
        return opt_kernels_[shard_id];
    }
}

void DenseTable::Save(std::string filepath) const {
    butil::Timer timer(butil::Timer::STARTED);

    int shard_id = PsCluster::Instance()->Rank();

    const auto& opt_kernel = GetOptKernels(shard_id);
    if (nullptr == opt_kernel) {
        return;
    }

    std::string file = filepath + "/dense_table/" + std::to_string(GetHandle())
                            + "/" + std::to_string(shard_id);

    FileWriterSink writer_sink(file);

    boost::iostreams::stream<FileWriterSink> out_stream(writer_sink);

    out_stream << "total_elements:" << total_elements_ << std::endl;
    out_stream << "rank_num:" << PsCluster::Instance()->RankNum() << std::endl;

    opt_kernel->Serialized(out_stream);

    out_stream.flush();

    timer.stop();

    LOG(INFO) << "DenseTable save, shard_id:" << shard_id
        << " size:" << opt_kernel->DataSize()
        << " latency:" << timer.s_elapsed() << "s";
}

void DenseTable::Load(std::string filepath) {
    butil::Timer timer(butil::Timer::STARTED);

    int shard_id = PsCluster::Instance()->Rank();
    const auto& opt_kernel = GetOptKernels(shard_id);
    if (nullptr == opt_kernel) {
        return;
    }

    std::string file = filepath + "/dense_table/" + std::to_string(GetHandle())
                            + "/" + std::to_string(shard_id);

    FileReaderSource reader_source(file);
    boost::iostreams::stream<FileReaderSource> in_stream(reader_source);

    int rank_num = 0;
    in_stream.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> total_elements_;
    in_stream.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> rank_num;

    CHECK_EQ(PsCluster::Instance()->RankNum(), rank_num);

    CHECK_EQ(0, Init(total_elements_));
    opt_kernel->DeSerialized(in_stream);

    timer.stop();

    LOG(INFO) << "DenseTable load, shard_id:" << shard_id
        << " size:" << opt_kernel->DataSize()
        << " latency:" << timer.s_elapsed() << "s";
}

DenseTableRegistry* DenseTableRegistry::Instance() {
    static DenseTableRegistry singleton;
    return &singleton;
}

uint32_t DenseTableRegistry::Register(DenseTable* table) {
    const std::lock_guard<std::mutex> lock(table_mu_);

    int table_handle = tables_.size();

    tables_.push_back(table);

    return table_handle;
}

DenseTable* DenseTableRegistry::Get(uint32_t table_handle) {
    CHECK_LT(table_handle, tables_.size()) << " table_handle:" << table_handle
        << " table size:" <<tables_.size();

    return tables_[table_handle];
}

DenseTable* CreateDenseTable(const OptimizerBase* opt) {
    DenseTable* table = new DenseTable(opt);

    table->SetHandle(DenseTableRegistry::Instance()->Register(table));

    return table;
}

} // namespace tensornet {
