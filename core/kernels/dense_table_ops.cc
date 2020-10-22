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

#include "core/utility/semaphore.h"
#include "core/ps/table/dense_table.h"
#include "core/ps/ps_cluster.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"

#include "core/kernels/resource_var_wrapper.h"
#include "core/ps/optimizer/optimizer_kernel.h"

#include <brpc/controller.h>

using namespace tensornet;

namespace tensorflow {

static void NoOpDeleter(void *) {}

template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      T** value);

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);

class DensePushPullCall {
public:
    DensePushPullCall(int table_handle, int shard_id)
        : shard_id_(shard_id) {
        req.set_table_handle(table_handle);
    }

    ~DensePushPullCall() {}

    void AddRequestData(butil::IOBuf& k_buf) {
        butil::IOBuf &buf = cntl.request_attachment();
        buf.append(k_buf);
    }

    void Start(const tensornet::Callback& done) {
        cntl.http_request().set_method(brpc::HTTP_METHOD_POST);

        const PsServerInterface* si =
            PsCluster::Instance()->GetServer(shard_id_);
        si->DensePushPullAsync(&cntl, &req, &resp, done);
    }

public:
    brpc::Controller cntl;
    DensePushPullRequest req;
    DensePushPullResponse resp;

private:
    int shard_id_ = -1;
};

class DenseTableInitKernel : public OpKernel {
public:
    explicit DenseTableInitKernel(OpKernelConstruction* c)
        : OpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void Compute(OpKernelContext* c) override {
        butil::IOBuf initial_value_buf;
        int total_elements = 0;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle &handle = HandleFromInput(c, i);

            Var *variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK(c, status);
            CHECK(variable);
            Tensor *var_tensor = variable->tensor();

            total_elements += var_tensor->NumElements();

            initial_value_buf.append_user_data(var_tensor->flat<float>().data(),
                                 var_tensor->NumElements() * sizeof(float),
                                  NoOpDeleter);
        }

        DenseTable* table = DenseTableRegistry::Instance()->Get(table_handle_);

        OP_REQUIRES(c, nullptr != table,
                errors::InvalidArgument("DenseTable have not created yet, handle:",
                    table_handle_));

        OP_REQUIRES(c, 0 == table->Init(total_elements),
                errors::InvalidArgument("DenseTable Init fail, total_element:",
                    total_elements));

        OP_REQUIRES(c, 0 == table->SetWeight(initial_value_buf),
                errors::InvalidArgument("DenseTable Init SetWeight fail"));

        return;
    }

private:
    int table_handle_;
    int N_;
};

REGISTER_KERNEL_BUILDER(Name("DenseTableInit").Device(DEVICE_CPU),
                        DenseTableInitKernel);

class DenseTablePushPullKernel : public AsyncOpKernel {
public:
    explicit DenseTablePushPullKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        OpInputList grads;
        OP_REQUIRES_OK_ASYNC(c, c->input_list("grads", &grads), done);

        OP_REQUIRES_ASYNC(c, c->num_inputs() == N_ * 2,
                          errors::InvalidArgument("DenseTable pushpull num_inputs:",
                                                  c->num_inputs(),
                                                  " not equal:", N_ * 2),
                          done);

        std::vector<Var*> variables;

        int total_elements = 0;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle &handle = HandleFromInput(c, i);
            const Tensor& grad_tensor = grads[i];

            Var *variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK_ASYNC(c, status, done);
            CHECK(variable);
            Tensor *var_tensor = variable->tensor();

            OP_REQUIRES_ASYNC(c, var_tensor->NumElements() == grad_tensor.NumElements(),
                              errors::InvalidArgument("DenseTable var tensor length:",
                                                      var_tensor->NumElements(),
                                                      " not equal grad tensor length:",
                                                      grad_tensor.NumElements()),
                              done);

            total_elements += grad_tensor.NumElements();
            variables.push_back(variable);
        }

        butil::IOBuf buf;

        for (int i = 0; i < N_; ++i) {
            const Tensor& grad_tensor = grads[i];
            const float* grad_data = grad_tensor.flat<float>().data();
            buf.append_user_data(const_cast<float *>(grad_data),
                                 grad_tensor.NumElements() * sizeof(float),
                                  NoOpDeleter);
        }

        DenseTable* table = DenseTableRegistry::Instance()->Get(table_handle_);

        OP_REQUIRES_ASYNC(c, nullptr != table,
                          errors::InvalidArgument("DenseTable not found:",
                                                  table_handle_),
                          done);

        CHECK_EQ(total_elements, table->TotalElements());

        int shard_num = PsCluster::Instance()->RankNum();
        Semaphore semaphore(shard_num);

        for (int shard_id = 0; shard_id < shard_num; shard_id++) {
            const auto* opt_kernel = table->GetOptKernels(shard_id).get();

            if (nullptr == opt_kernel) {
                semaphore.Notify();
                continue;
            }

            auto* call = new DensePushPullCall(table_handle_, shard_id);

            butil::IOBuf k_buf;
            int k_len = opt_kernel->Length() * sizeof(float);
            CHECK_EQ(k_len, buf.cutn(&k_buf, k_len));
            call->AddRequestData(k_buf);

            call->Start([call, variables, opt_kernel, k_len, &semaphore]() {
                std::unique_ptr<DensePushPullCall> call_free_guard(call);

                butil::IOBuf& output = call->cntl.response_attachment();

                CHECK_EQ(output.size(), k_len);

                for (int i = 0, offset = 0; i < (int)variables.size(); ++i) {
                    Var* variable = variables[i];

                    Tensor *var_tensor = variable->tensor();
                    float* var_data = var_tensor->flat<float>().data();

                    int num_elements = var_tensor->NumElements();

                    // find the first variable to populate
                    if (offset + num_elements <= opt_kernel->OffsetBegin()) {
                        offset += num_elements;
                        continue;
                    }

                    int var_offset = 0;

                    // fist populate variable may be not start with 0
                    if (opt_kernel->OffsetBegin() > offset) {
                        var_offset = opt_kernel->OffsetBegin() - offset;
                    }

                    CHECK_LT(var_offset, num_elements);

                    int copy_len = std::min(output.size(), (num_elements - var_offset) * sizeof(float));

                    butil::IOBuf copy_buf;
                    CHECK_EQ(copy_len, output.cutn(&copy_buf, copy_len));

                    copy_buf.copy_to(var_data + var_offset, copy_len);

                    if (output.size() > 0) {
                        offset += num_elements;
                    } else {
                        break;
                    }
                }

                semaphore.Notify();
            });
        }

        semaphore.WaitForSemaphore();

        done();

        return;
    }

private:
    int table_handle_;
    int N_;
};

REGISTER_KERNEL_BUILDER(Name("DenseTablePushPull").Device(DEVICE_CPU),
                        DenseTablePushPullKernel);

}  // namespace tensorflow
