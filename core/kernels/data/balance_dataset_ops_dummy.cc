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

#include "core/kernels/data/balance_dataset_ops_dummy.h"

#include "core/public/version.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const BalanceDatasetOp::kDatasetType;
/* static */ constexpr const char* const BalanceDatasetOp::kInputDataset;
/* static */ constexpr const char* const BalanceDatasetOp::kOutputTypes;
/* static */ constexpr const char* const BalanceDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kBalanceDataset[] = "BalanceDataset";

class BalanceDatasetOp::Dataset : public DatasetBase {
public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx))
        , input_(input) {
        input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
        data::name_utils::IteratorPrefixParams params;
        return absl::make_unique<Iterator>(Iterator::Params{
            this, data::name_utils::IteratorPrefix(kDatasetType, prefix, params)});
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

    int64 Cardinality() const override {
        return input_->Cardinality();
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
            mutex_lock l(mu_);

            if (!input_impl_) {
                // TODO Get data from brpc

                *end_of_sequence = false;
                return Status::OK();
            }

            TF_RETURN_IF_ERROR(
                    input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
            *end_of_sequence = false;
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

    private:
        mutex mu_;
        std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
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
