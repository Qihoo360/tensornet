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

#include "core/kernels/data/balance_dataset_ops.h"

#include "core/public/version.h"
#include "core/utility/semaphore.h"

#include <brpc/server.h>
#include <butil/rand_util.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"

using namespace tensornet;

namespace tensorflow {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const BalanceDatasetOp::kDatasetType;
/* static */ constexpr const char* const BalanceDatasetOp::kInputDataset;
/* static */ constexpr const char* const BalanceDatasetOp::kOutputTypes;
/* static */ constexpr const char* const BalanceDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kBalanceDataset[] = "BalanceDataset";

class BalanceDataCall {
public:
    BalanceDataCall(uint32_t shard_id, uint32_t balance_handle)
        : shard_id_(shard_id) {
        req.set_req_shard_id(PsCluster::Instance()->Rank());
        req.set_balance_handle(balance_handle);
    }

    ~BalanceDataCall() {}

    void Start(const tensornet::Callback& done) {
        const PsServerInterface* si =
            PsCluster::Instance()->GetServer(shard_id_);
        si->DatasetPullAsync(&cntl, &req, &resp, done);
    }

public:
    brpc::Controller cntl;
    DatasetPullRequest req;
    DatasetPullResponse resp;

private:
    uint32_t shard_id_ = -1;
};

void BalanceInputDataInfo::ProcessBrpcDatasetPullReq(const DatasetPullRequest* req, DatasetPullResponse* resp) {
    resp->set_resp_shard_id(PsCluster::Instance()->Rank());

    ChangeShardStatus(req->req_shard_id());

    if (GetFinished()) {
        resp->set_end_of_sequence(true);
        return;
    }

    uint32_t balance_handle = req->balance_handle();

    CHECK(op_elements_.count(balance_handle)) << "balance_handle " << balance_handle << " not registered.";
    auto* elements = op_elements_[balance_handle];
    std::vector<Tensor> tensors;
    if (elements->get(&tensors)) {
        VariantTensorData variant_tensor;
        {
            for (auto& element : tensors) {
                *(variant_tensor.add_tensors()) = element;
            }
        }
        resp->set_end_of_sequence(false);
        resp->set_dataset_info(variant_tensor.SerializeAsString());
    } else {
        resp->set_end_of_sequence(GetFinished());
    }
}

void BalanceInputDataInfo::SendBrpcDatasetPullReq(uint32_t balance_handle, bool* no_shard_remaining) {
    std::vector<BalanceDataCall*> calls;
    {
        const std::lock_guard<std::mutex> lock(RemainingShardsMutex());
        for (auto shard : *RemainingShards()) {
            calls.emplace_back(new BalanceDataCall(shard, balance_handle));
        }
    }

    if (calls.empty()) {
        *no_shard_remaining = true;
    } else {
        *no_shard_remaining = false;
    }

    Semaphore semaphore(calls.size());
    for (auto& call : calls) {
        call->Start([this, call, &semaphore, balance_handle]() {
            this->CopyDataToBuffer(&(call->resp), balance_handle);
            semaphore.Notify();
            delete call;
        });
    }

    semaphore.WaitForSemaphore();
}

void BalanceInputDataInfo::CopyDataToBuffer(const DatasetPullResponse* resp, uint32_t balance_handle) {
    if (resp->dataset_info().length() == 0) {
        if (resp->end_of_sequence()) {
            ChangeShardStatus(resp->resp_shard_id());
        }
        return;
    }
    VariantTensorData variant_tensor;
    variant_tensor.ParseFromString(resp->dataset_info());
    BufferQueueWithLock* q = op_elements_[balance_handle];
    std::vector<Tensor> brpc_data;
    for (const Tensor& tensor : variant_tensor.tensors()) {
        brpc_data.emplace_back(std::move(tensor));
    }
    q->put(std::move(brpc_data));
}

class BalanceDatasetOp::Dataset : public DatasetBase {
public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx))
        , input_(input) {
        input_->Ref();
        auto* data_info = BalanceInputDataInfo::Instance();
        balance_handle_ = data_info->Register(&brpc_element_);
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
        return absl::make_unique<Iterator>(Iterator::Params{
            this, data::name_utils::IteratorPrefix(kDatasetType, prefix)});
    }

    const DataTypeVector& output_dtypes() const override {
        return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
        return input_->output_shapes();
    }

    string DebugString() const override {
        return data::name_utils::DatasetDebugString(kDatasetType);
    }

    Status CheckExternalState() const override {
        return input_->CheckExternalState();
    }

protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
        Node* input_graph_node = nullptr;
        TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
        TF_RETURN_IF_ERROR(
            b->AddDataset(this, {input_graph_node}, output));
        return Status::OK();
    }

private:
    class Iterator : public DatasetIterator<Dataset> {
    public:
        explicit Iterator(const Params& params)
            : DatasetIterator<Dataset>(params) {}
            Status Initialize(IteratorContext* ctx) override {
            return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
            {
                mutex_lock l(mu_);
                if (!input_impl_) {
                    *end_of_sequence = true;
                    return Status::OK();
                }
            }

            bool has_data = false;
            TF_RETURN_IF_ERROR(GetNextManyInternal(ctx, end_of_sequence, out_tensors, &has_data));

            auto* data_info = BalanceInputDataInfo::Instance();

            if (*end_of_sequence) {
                data_info->SetFinished(true);
            }

            if (has_data) {
                *end_of_sequence = false;
                return Status::OK();
            }

            BufferQueueWithLock* q = data_info->op_elements_[dataset()->balance_handle_];
            if (q->empty() && *end_of_sequence) {
                GetDataFromBrpcInternal(end_of_sequence, out_tensors);
                return Status::OK();
            }

            if (!q->empty() || !*end_of_sequence) {
                q->get(out_tensors);
                *end_of_sequence = false;
            }


            return Status::OK();
        }

    protected:
        std::shared_ptr<data::model::Node> CreateNode(
                IteratorContext* ctx, data::model::Node::Args args) const override {
            return data::model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
        }

#if defined(TN_COMPATIBLE_INTERFACE_2_2)
        Status SaveInternal(IteratorStateWriter* writer) override {
            mutex_lock l(mu_);
            if (!input_impl_) {
                TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
            } else {
                TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
            }
            return Status::OK();
        }
#else 
        Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
            mutex_lock l(mu_);
            if (!input_impl_) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
            } else {
              TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
            }
            return Status::OK();
        }
#endif

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
            mutex_lock l(mu_);
            if (!reader->Contains(full_name(kInputImplEmpty))) {
                TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
            } else {
                input_impl_.reset();
            }
            return Status::OK();
        }

        Status GetNextManyInternal(IteratorContext* ctx,
                                   bool* end_of_sequence,
                                   std::vector<Tensor>* out_tensors,
                                   bool* has_data) {
            mutex_lock l(mu_);

            TF_RETURN_IF_ERROR(
                    input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
            if (*end_of_sequence) {
                *has_data = false;
                return Status::OK();
            }
            *has_data = true;

            auto* data_info = BalanceInputDataInfo::Instance();
            BufferQueueWithLock* q = data_info->op_elements_[dataset()->balance_handle_];
            while (!q->buffer_full() && !*end_of_sequence) {
                std::vector<Tensor> input_vec;
                TF_RETURN_IF_ERROR(
                    input_impl_->GetNext(ctx, &input_vec, end_of_sequence));
                if (!*end_of_sequence) {
                    q->put(std::move(input_vec));
                }
            }

            return Status::OK();
        }

        void GetDataFromBrpcInternal(bool* end_of_sequence, std::vector<Tensor>* out_tensors) {
            auto* data_info = BalanceInputDataInfo::Instance();
            BufferQueueWithLock* q = data_info->op_elements_[dataset()->balance_handle_];
            bool no_shard = false;
            while (!no_shard) {
                data_info->SendBrpcDatasetPullReq(dataset()->balance_handle_, &no_shard);
                if (q->get(out_tensors)) {
                    *end_of_sequence = false;
                    return;
                }
            }
            *end_of_sequence = true;
        }

    private:
        mutex mu_;
        std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
        bool first_brpc_req_ = true;
    };

    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;

    uint32_t balance_handle_;
    BufferQueueWithLock brpc_element_;
};

BalanceDatasetOp::BalanceDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
}

void BalanceDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
    *output = new Dataset(ctx, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("BalanceDataset").Device(DEVICE_CPU),
                        BalanceDatasetOp);
}  // namespace
}  // namespace tensorflow
