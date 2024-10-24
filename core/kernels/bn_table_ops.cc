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
#include "core/ps/table/bn_table.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"

#include "core/kernels/resource_var_wrapper.h"
#include "core/ps_interface/ps_raw_interface.h"


#include <brpc/controller.h>
#include <sstream>
#include <Eigen/Dense>
#include <iostream>
#include <mutex>

#include "core/ps/ps_server_interface.h"
#include "core/ps/ps_cluster.h"

using namespace tensornet;

namespace tensorflow {

static void NoOpDeleter(void *) {}

template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);

class BnStatisticsPushCall {
public:
    BnStatisticsPushCall(int table_handle, int shard_id)
        : shard_id_(shard_id) {
		req.set_req_shard_id(shard_id);
        req.set_table_handle(table_handle);
    }

    ~BnStatisticsPushCall() {}

    void AddRequestData(butil::IOBuf& k_buf) {
        butil::IOBuf &buf = cntl.request_attachment();
        buf.append(k_buf);
    }

    void Start(const tensornet::Callback& done) {
		const PsServerInterface* si = 
			PsCluster::Instance()->GetServer(shard_id_);
        si->BnStatisticsPushAsync(&cntl, &req, &resp, done);
    }

public:
    brpc::Controller cntl;
    BnStatisticsPushRequest req;
    BnStatisticsPushResponse resp;

private:
    int shard_id_ = -1;
};


class BnStatisticsPushKernel : public AsyncOpKernel {
public:
    explicit BnStatisticsPushKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
        OP_REQUIRES_OK(c, c->GetAttr("synchronized", &synchronized_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        butil::IOBuf acc_buf; 

        std::vector<double*> allocated_pointers;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle& handle = HandleFromInput(c, i);

            Var* variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK_ASYNC(c, status, done);
            CHECK(variable);

            Tensor *var_tensor = variable->tensor();

            int num_elements = var_tensor->NumElements();
            double* dynamic_double_data = new double[num_elements];
            const float* float_data = var_tensor->flat<float>().data();
            for (int i = 0; i < num_elements; ++i) {
                dynamic_double_data[i] = static_cast<double>(float_data[i]);    
            }
            acc_buf.append_user_data(dynamic_double_data, num_elements * sizeof(double), NoOpDeleter);
            allocated_pointers.push_back(dynamic_double_data);
        }

        BnTable* table = BnTableRegistry::Instance()->Get(table_handle_);
        table->Append(acc_buf, true);

        for (auto ptr : allocated_pointers) {  
            delete[] ptr;  
        }  
        allocated_pointers.clear();

        if(synchronized_){
			PsCluster* cluster = PsCluster::Instance();
			OP_REQUIRES_ASYNC( c, true == cluster->IsInitialized(),
            errors::InvalidArgument("cluster instance not initialized:"), done);

			butil::IOBuf inc_buf;
			table->GetIncStatistics(inc_buf);

			std::vector<BnStatisticsPushCall*> calls;

			for (size_t shard_id = 0; shard_id < cluster->RankNum(); shard_id++) {
				if(shard_id != cluster->Rank()){
				    auto* call = new BnStatisticsPushCall(table_handle_, shard_id);
				    call->AddRequestData(inc_buf);
				    calls.emplace_back(call);
				}
			}

			Semaphore semaphore(calls.size());

			for (auto& call : calls) {
				call->Start([this, call, &semaphore]() {
                    semaphore.Notify();
					delete call;
					});
			}

			semaphore.WaitForSemaphore();
        }

        done();
        
        return;
    }

private:
    int table_handle_;
    int N_;
    bool synchronized_;
};

REGISTER_KERNEL_BUILDER(Name("BnStatisticsPush").Device(DEVICE_CPU),
                        BnStatisticsPushKernel);

class UpdateMomentsKernel : public OpKernel {
public:
    explicit UpdateMomentsKernel(OpKernelConstruction* c)
        : OpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void Compute(OpKernelContext* c) override {
        std::vector<Var*> bn_vars;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle &handle = HandleFromInput(c, i);

            Var *variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK(c, status);
            CHECK(variable);
            bn_vars.emplace_back(variable);
        }

        BnTable* table = BnTableRegistry::Instance()->Get(table_handle_);

        std::tuple<Eigen::ArrayXf, Eigen::ArrayXf> moments_tuple = table->GetMoments();

        auto& global_mean_var = bn_vars[0];
        float* global_mean_flat = global_mean_var->tensor()->flat<float>().data();
        std::copy(std::get<0>(moments_tuple).data(), std::get<0>(moments_tuple).data() + std::get<0>(moments_tuple).size(), global_mean_flat);

        auto& global_var_var = bn_vars[1];
        float* global_var_flat = global_var_var->tensor()->flat<float>().data();
        std::copy(std::get<1>(moments_tuple).data(), std::get<1>(moments_tuple).data() + std::get<1>(moments_tuple).size(), global_var_flat);
        
        return;
    }

private:
    int table_handle_;
    int N_;
};


REGISTER_KERNEL_BUILDER(Name("UpdateMoments").Device(DEVICE_CPU),
                        UpdateMomentsKernel);

class BnStatisticsPullCall {
public:
    BnStatisticsPullCall(int table_handle, int shard_id)
        : shard_id_(shard_id) {
		req.set_req_shard_id(shard_id);
        req.set_table_handle(table_handle);
    }

    ~BnStatisticsPullCall() {}

    void Start(const tensornet::Callback& done) {
		const PsServerInterface* si = 
			PsCluster::Instance()->GetServer(shard_id_);
        si->BnStatisticsPullAsync(&cntl, &req, &resp, done);
    }

public:
    brpc::Controller cntl;
    BnStatisticsPullRequest req;
    BnStatisticsPullResponse resp;

private:
    int shard_id_ = -1;
};


class BnStatisticsPullKernel : public AsyncOpKernel {
public:
    explicit BnStatisticsPullKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {

        std::vector<Var*> bn_vars;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle &handle = HandleFromInput(c, i);

            Var *variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK(c, status);
            CHECK(variable);
            bn_vars.emplace_back(variable);
        }

        PsCluster* cluster = PsCluster::Instance();
        OP_REQUIRES_ASYNC(
            c, true == cluster->IsInitialized(),
            errors::InvalidArgument("cluster instance not initialized:"), done);

        BnTable *table = BnTableRegistry::Instance()->Get(table_handle_);
        std::vector<BnStatisticsPullCall*> calls;

        for (size_t shard_id = 0; shard_id < cluster->RankNum(); shard_id++) {
            if(shard_id != cluster->Rank()){
            calls.emplace_back(
                new BnStatisticsPullCall(table_handle_, shard_id));
            }
        }

        Semaphore semaphore(calls.size());

        for (auto& call : calls) {
            call->Start([this, call, &table, &semaphore]() {
				table->Append(call->cntl.response_attachment(), false);
                semaphore.Notify();
                delete call;
            });
        }

        semaphore.WaitForSemaphore();
        std::tuple<Eigen::ArrayXf, Eigen::ArrayXf> moments_tuple = table->GetMoments();

        auto& global_mean_var = bn_vars[0];
        float* global_mean_flat = global_mean_var->tensor()->flat<float>().data();
        std::copy(std::get<0>(moments_tuple).data(), std::get<0>(moments_tuple).data() + std::get<0>(moments_tuple).size(), global_mean_flat);

        auto& global_var_var = bn_vars[1];
        float* global_var_flat = global_var_var->tensor()->flat<float>().data();
        std::copy(std::get<1>(moments_tuple).data(), std::get<1>(moments_tuple).data() + std::get<1>(moments_tuple).size(), global_var_flat);

        done();

        return;
    }

private:
    int table_handle_;
    int N_;
};

REGISTER_KERNEL_BUILDER(Name("BnStatisticsPull").Device(DEVICE_CPU),
                        BnStatisticsPullKernel);

};
