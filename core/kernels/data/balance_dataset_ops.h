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
#include <condition_variable>
#include <chrono>

#include <butil/time.h>

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
        cv_.notify_one();
    }

    void put(std::vector<std::vector<Tensor> >&& elements) {
        const std::lock_guard<std::mutex> lock(mu_);
        for (auto&& v : elements) {
            elements_.emplace(v);
        }
        cv_.notify_all();
        
    }

    bool empty() {
        const std::lock_guard<std::mutex> lock(mu_);
        return elements_.empty();
    }

    size_t fill_count() {
        const std::lock_guard<std::mutex> lock(mu_);
        return buffer_size_ - elements_.size();
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

    bool get_wait(std::vector<Tensor>* tensors) {
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (empty_unlock()) {
                cv_.wait_for(lock, std::chrono::seconds(timeout_s_));
            }
        }

        return get(tensors);
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
    int64_t timeout_s_ = 10;
    std::condition_variable cv_;
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

        timer_.start();

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

    double TimerElapsedInSecond() {
        timer_.stop();
        return timer_.s_elapsed();
    }

public:
    std::mutex remaining_mu_;
    std::set<uint32_t> remaining_shards_;

    std::mutex mu_;
    bool finished_ = false;
    std::map<uint32_t, BufferQueueWithLock*> op_elements_;

    butil::Timer timer_;
};

}  // namespace tensorflow

#endif  // TENSORNET_CORE_KERNELS_DATA_BALANCE_DATASET_OP_H_
