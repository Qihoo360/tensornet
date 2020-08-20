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

#ifndef TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_
#define TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

#include <set>
#include <vector>
#include <mutex>
#include <queue>

#include "core/ps/ps_server_interface.h"
#include "core/ps/ps_cluster.h"

namespace tensorflow {

class BalanceDatasetOp : public tensorflow::UnaryDatasetOpKernel {
public:
    static constexpr const char* const kDatasetType = "Balance";
    static constexpr const char* const kInputDataset = "input_dataset";
    static constexpr const char* const kOutputTypes = "output_types";
    static constexpr const char* const kOutputShapes = "output_shapes";

    explicit BalanceDatasetOp(tensorflow::OpKernelConstruction* ctx);

protected:
    void MakeDataset(tensorflow::OpKernelContext* ctx, tensorflow::DatasetBase* input,
                    tensorflow::DatasetBase** output) override;

private:
    class Dataset;
};

class BufferQueueWithLock {
public:
    void put(std::vector<Tensor>&& element) {
        const std::lock_guard<std::mutex> lock(mu_);
        elements_.emplace(element);
    }

    bool empty() {
        const std::lock_guard<std::mutex> lock(mu_);
        return elements_.empty();
    }

    bool buffer_full() {
        const std::lock_guard<std::mutex> lock(mu_);
        return elements_.size() > buffer_size_;
    }

    bool get(std::vector<Tensor>* tensors) {
        const std::lock_guard<std::mutex> lock(mu_);

        if (empty_unlock()) {
            return false;
        }
        
        *tensors = std::move(elements_.front());
        elements_.pop();
        return true;
    }

    void pop() {
        const std::lock_guard<std::mutex> lock(mu_);
        elements_.pop();
    }

    size_t size() {
        const std::lock_guard<std::mutex> lock(mu_);
        return elements_.size();
    }

private:
    bool empty_unlock() {
        return elements_.empty();
    }

private:
    size_t buffer_size_ = 100;
    std::mutex mu_;
    std::queue<std::vector<Tensor> > elements_;
};

class BalanceInputDataInfo {
public:
    static BalanceInputDataInfo* Instance() {
        static BalanceInputDataInfo instance;
        return &instance;
    }

    uint32_t Register(BufferQueueWithLock* elements) {
        const std::lock_guard<std::mutex> lock(mu_);
        uint32_t handle = op_elements_.size();
        // LOG(INFO) << "Register:" << handle << " pid:" << std::this_thread::get_id();
        op_elements_[handle] = elements;
        return handle;
    }

    int Init(size_t shard_num, int self_shard) {
        const std::lock_guard<std::mutex> lock(remaining_mu_);
        for (size_t i = 0; i < shard_num; ++i) {
            remaining_shards_.insert(i);
        }
        remaining_shards_.erase(self_shard);

        finished_ = false;

        return 0;
    }

    const std::set<uint32_t>* RemainingShards() {
        return &remaining_shards_;
    }

    std::mutex& RemainingShardsMutex() {
        return remaining_mu_;
    }

    void ChangeShardStatus(uint32_t shard_id) {
        const std::lock_guard<std::mutex> lock(remaining_mu_);
        remaining_shards_.erase(shard_id);
    }

    bool GetFinished() { return finished_; }

    void SetFinished(bool finished) { finished_ = finished; }

    void ProcessBrpcDatasetPullReq(const tensornet::DatasetPullRequest* req, tensornet::DatasetPullResponse* resp);

    void SendBrpcDatasetPullReq(uint32_t balance_handle, bool* no_shard_remaining);

    void CopyDataToBuffer(const tensornet::DatasetPullResponse* resp, uint32_t balance_handle);

public:
    std::mutex remaining_mu_;
    std::set<uint32_t> remaining_shards_;

    std::mutex mu_;
    bool finished_ = false;
    std::map<uint32_t, BufferQueueWithLock*> op_elements_;
};

}  // namespace tensorflow

#endif  // TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_
