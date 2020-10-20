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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"

#include "core/kernels/resource_var_wrapper.h"
#include "core/ps_interface/ps_raw_interface.h"

#include <brpc/controller.h>
#include <sstream>

#include "core/ps/ps_server_interface.h"
#include "core/ps/ps_cluster.h"

using namespace tensornet;

namespace tensorflow {

template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);

class SparsePullCall {
public:
    SparsePullCall(int table_handle, int shard_id, int dim)
        : shard_id_(shard_id) {
        req.set_table_handle(table_handle);
        req.set_dim(dim);
    }

    ~SparsePullCall() {}

    void AddRequestSign(size_t var_index, size_t sign_index, uint64 sign) {
        req.add_signs(sign);

        call_sign_infos.emplace_back(var_index, sign_index);
    }

    void Start(const tensornet::Callback& done) {
        if (call_sign_infos.empty()) {
            done();
        } else {
            const PsServerInterface* si =
                PsCluster::Instance()->GetServer(shard_id_);
            si->SparsePullAsync(&cntl, &req, &resp, done);
        }
    }

public:
    brpc::Controller cntl;
    SparsePullRequest req;
    SparsePullResponse resp;

    std::vector<std::pair<size_t, size_t>> call_sign_infos;

private:
    int shard_id_ = -1;
};

class SparsePushCall {
public:
    SparsePushCall(int table_handle, int shard_id, int dim)
        : shard_id_(shard_id) {
        req.set_table_handle(table_handle);
        req.set_dim(dim);
    }

    ~SparsePushCall() {}

    void AddRequestGrad(const SparsePushSignInfo& sign_info, const float* grad_vec, int dim) {
        butil::IOBuf &buf = cntl.request_attachment();
        buf.append(&sign_info, sizeof(sign_info));
        buf.append(grad_vec, dim * sizeof(float));
    }

    void Start(const tensornet::Callback& done) {
        butil::IOBuf &buf = cntl.request_attachment();
        if (buf.size() <= 0) {
            done();
        } else {
            const PsServerInterface* si =
                PsCluster::Instance()->GetServer(shard_id_);
            si->SparsePushAsync(&cntl, &req, &resp, done);
        }
    }

public:
    brpc::Controller cntl;
    SparsePushRequest req;
    SparsePushResponse resp;

private:
    int shard_id_ = -1;
};

struct SparsePullVarInfo {
public:
    SparsePullVarInfo(tensorflow::Var* t_var,
                      const tensorflow::Tensor* value, Tensor* out_tensor)
        : var(t_var)
        , sign_value(value) 
        , out_tensor(out_tensor) {
        const int64* feasign_vec = value->flat<int64>().data();
        int64* out_vec = out_tensor->flat<int64>().data();

        std::map<uint64, size_t> sign_id_mapping;
        for (int i = 0; i < value->NumElements(); ++i) {
            const uint64 sign = (uint64)feasign_vec[i];
            auto inserted = sign_id_mapping.insert({sign, sign_id_mapping.size()});
            if (inserted.second) {
                signs.push_back(sign);
            }

            out_vec[i] = inserted.first->second;
        }

        const Tensor* var_tensor = var->tensor();

        const uint64 max_var_count = var_tensor->shape().dim_size(0);
        CHECK_LT(signs.size(), max_var_count);
    }

    int VarDim() const {
        const Tensor* var_tensor = var->tensor();
        return var_tensor->shape().dim_size(1);
    }

public:
    // sparse feature column embedding variable.
    // shape: [max_var_count, emb_dim]
    tensorflow::Var* var;

    // sign list of feature column
    // shape: [sign_count]
    // its store feature signs of one feature_column in one batch
    const tensorflow::Tensor* sign_value;

    Tensor* out_tensor;

    // in order to use embedding_lookup, we must map the feature sign into index
    // which will not exceed *var* first dimension(max_var_count)
    // the mapping id begin from 0, depend on signs order in sign_value 
    std::vector<uint64> signs;
};

class SparseTablePullKernel : public AsyncOpKernel {
public:
    explicit SparseTablePullKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        OP_REQUIRES_ASYNC(c, c->num_inputs() == N_ * 2,
                          errors::InvalidArgument("SparseTable pull num_inputs:",
                                                  c->num_inputs(),
                                                  " not equal:", N_ * 2),
                          done);
        std::vector<SparsePullVarInfo> var_infos;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle& handle = HandleFromInput(c, i);

            Var* variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK_ASYNC(c, status, done);
            CHECK(variable);

            const Tensor* var_tensor = variable->tensor();
            Tensor* out_tensor = nullptr;
            const Tensor* sign_value = &c->input(N_ + i);

            OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, sign_value->shape(), &out_tensor), done);

            OP_REQUIRES_ASYNC(
                c, TensorShapeUtils::IsMatrix(var_tensor->shape()),
                errors::InvalidArgument(
                    "sparse pull variable must Matrix(sign_id_cnt, dim), saw: ",
                    var_tensor->shape().DebugString()),
                done);

            var_infos.emplace_back(variable, sign_value, out_tensor);
        }

        CHECK_GT(var_infos.size(), 0);

        int dim = var_infos[0].VarDim();

        for (size_t i = 0; i < var_infos.size(); i++) {
            CHECK_EQ(dim, var_infos[i].VarDim());
        }

        PsCluster* cluster = PsCluster::Instance();
        OP_REQUIRES_ASYNC(
            c, true == cluster->IsInitialized(),
            errors::InvalidArgument("cluster instance not initialized:"), done);

        std::vector<SparsePullCall*> calls;

        for (size_t shard_id = 0; shard_id < cluster->RankNum(); shard_id++) {
            calls.emplace_back(
                new SparsePullCall(table_handle_, shard_id, dim));
        }

        for (size_t var_index = 0; var_index < var_infos.size(); var_index++) {
            for (size_t sign_index = 0; sign_index < var_infos[var_index].signs.size(); sign_index++) {
                const uint64 sign = var_infos[var_index].signs[sign_index];

                int shard_id = sign % cluster->RankNum();
                calls[shard_id]->AddRequestSign(var_index, sign_index, sign);
            }
        }

        Semaphore semaphore(calls.size());

        for (auto& call : calls) {
            call->Start([this, call, &var_infos, &semaphore]() {
                PopulatePulledVariable_(var_infos, call->call_sign_infos,
                    call->resp, call->cntl.response_attachment());
                semaphore.Notify();
                delete call;
            });
        }

        semaphore.WaitForSemaphore();

        done();

        return;
    }

private:
    void PopulatePulledVariable_(std::vector<SparsePullVarInfo>& var_infos,
                                   const std::vector<std::pair<size_t, size_t>>& call_sign_infos,
                                   const SparsePullResponse& resp, butil::IOBuf& emb_buf) {
        int dim = resp.dim();

        for (size_t i = 0; i < call_sign_infos.size(); i++) {
            size_t var_index = call_sign_infos[i].first;
            size_t sign_index = call_sign_infos[i].second;

            CHECK_LT(var_index, var_infos.size());

            auto& var_info = var_infos[var_index];
            Tensor* var_tensor = var_info.var->tensor();
            CHECK_EQ(dim, var_info.VarDim());

            float* w_matrix = var_tensor->matrix<float>().data();

            size_t emb_size = sizeof(float) * dim;
            CHECK_EQ(emb_size, emb_buf.cutn(w_matrix + sign_index * dim, emb_size));
        }
    }

private:
    int table_handle_;
    int N_;
};

REGISTER_KERNEL_BUILDER(Name("SparseTablePull").Device(DEVICE_CPU),
                        SparseTablePullKernel);

struct SparsePushVarInfo {
public:
    SparsePushVarInfo(const Tensor* t_value, const Tensor* t_grad)
        : value(t_value)
        , grad(t_grad) {

        const int64* feasign_vec = value->flat<int64>().data();

        std::map<uint64, int> sign_id_mapping;
        for (int i = 0; i < value->NumElements(); ++i) {
            uint64 sign = (uint64)feasign_vec[i];
            auto ret = sign_id_mapping.insert({sign, sign_id_mapping.size()});

            if (ret.second) {
                virtual_sign_infos.emplace_back(sign, 1);
            } else {
                auto iter = ret.first;
                virtual_sign_infos[iter->second].batch_show += 1;
            }
        }
    }

    int GradDim() const {
        return grad->shape().dim_size(1);
    }

public:
    const Tensor* value;
    const Tensor* grad;

    std::vector<SparsePushSignInfo> virtual_sign_infos;
};

class SparseTablePushKernel : public AsyncOpKernel {
public:
    explicit SparseTablePushKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        OP_REQUIRES_ASYNC(c, c->num_inputs() == N_ * 2,
                          errors::InvalidArgument("SparseTable push num_inputs:",
                                                  c->num_inputs(),
                                                  " not equal:", N_ * 2),
                          done);
        std::vector<SparsePushVarInfo> var_infos;

        for (int i = 0; i < N_; i++) {
            const Tensor* value = &c->input(i);
            const Tensor* grad = &c->input(N_ + i);

            OP_REQUIRES_ASYNC(
                c, TensorShapeUtils::IsMatrix(grad->shape()),
                errors::InvalidArgument(
                    "sparse push grad must Matrix(sign_id_cnt, dim), saw: ",
                    grad->shape().DebugString()),
                done);

            var_infos.emplace_back(value, grad);
        }

        CHECK_GT(var_infos.size(), 0);

        int dim = var_infos[0].GradDim();
        for (size_t i = 0; i < var_infos.size(); i++) {
            CHECK_EQ(dim, var_infos[i].GradDim());
        }

        std::vector<SparsePushCall*> calls;
        PsCluster* cluster = PsCluster::Instance();

        for (size_t shard_id = 0; shard_id < cluster->RankNum(); shard_id++) {
            calls.emplace_back(
                new SparsePushCall(table_handle_, shard_id, dim));
        }

        for (size_t i = 0; i < var_infos.size(); i++) {
            // NOTE, tensorfow use RowMajor layout
            const float* grad_matrix = var_infos[i].grad->matrix<float>().data();

            for (size_t sign_index = 0; sign_index < var_infos[i].virtual_sign_infos.size(); sign_index++) {
                const auto& sign_info = var_infos[i].virtual_sign_infos[sign_index];

                int shard_id = sign_info.sign % cluster->RankNum();
                const float* grad = grad_matrix + dim * sign_index;
                calls[shard_id]->AddRequestGrad(sign_info, grad, dim);
            }
        }

        for (auto& call : calls) {
            call->Start([this, call]() {
                delete call;
            });
        }

        done();
    }

private:
    int table_handle_;
    int N_;
};

REGISTER_KERNEL_BUILDER(Name("SparseTablePush").Device(DEVICE_CPU),
                        SparseTablePushKernel);

}  // namespace tensorflow
