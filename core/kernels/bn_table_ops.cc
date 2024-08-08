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

#include "core/ps/ps_server_interface.h"
#include "core/ps/ps_cluster.h"

using namespace tensornet;

namespace tensorflow {

static void NoOpDeleter(void *) {}

template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);

class BnVarsPullCall {
public:
    BnVarsPullCall(int table_handle, int shard_id)
        : shard_id_(shard_id) {
		req.set_req_shard_id(shard_id);
        req.set_table_handle(table_handle);
    }

    ~BnVarsPullCall() {}

    void Start(const tensornet::Callback& done) {
		const PsServerInterface* si = 
			PsCluster::Instance()->GetServer(shard_id_);
        si->BnVarsPullAsync(&cntl, &req, &resp, done);
    }

public:
    brpc::Controller cntl;
    BnVarsPullRequest req;
    BnVarsPullResponse resp;

private:
    int shard_id_ = -1;
};


class BnVarsInfos {
public:
	BnVarsInfos(int bn_size)
		: bn_size_(bn_size) {
	}

	void Add(butil::IOBuf& w_buf){
            CHECK_EQ( (bn_size_ * 2 + 1) * sizeof(float), w_buf.size());

	    Eigen::ArrayXf moving_mean_ = Eigen::ArrayXf::Constant(bn_size_, 1.0f);
	    w_buf.cutn(moving_mean_.data(), bn_size_ * sizeof(float));
	    moving_means_.emplace_back(moving_mean_);

	    Eigen::ArrayXf moving_var_ = Eigen::ArrayXf::Constant(bn_size_, 1.0f);
	    w_buf.cutn(moving_var_.data(), bn_size_ * sizeof(float));
	    moving_vars_.emplace_back(moving_var_);
     
            Eigen::ArrayXf batch_count_(1);
	    w_buf.cutn(batch_count_.data(), sizeof(float));
	    batch_counts_.emplace_back(batch_count_);
	}

	~BnVarsInfos() {}

        std::vector<Eigen::ArrayXf>& GetMovingMeans() {
            return moving_means_;
        }

        std::vector<Eigen::ArrayXf>& GetMovingVars() {
            return moving_vars_;
        }

        std::vector<Eigen::ArrayXf>& GetBatchCounts() {
            return batch_counts_;
        }

public:
	std::vector<Eigen::ArrayXf> moving_means_;
	std::vector<Eigen::ArrayXf> moving_vars_;
	std::vector<Eigen::ArrayXf> batch_counts_;
        int bn_size_;
};


class BnVarsPullKernel : public AsyncOpKernel {
public:
    explicit BnVarsPullKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        std::vector<Var*> bn_vars;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle& handle = HandleFromInput(c, i);

            Var* variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK_ASYNC(c, status, done);
            CHECK(variable);

            bn_vars.emplace_back(variable);
        }

        CHECK_EQ(bn_vars.size(), 3);

		uint32_t bn_size = bn_vars.front()->tensor()->NumElements();

        PsCluster* cluster = PsCluster::Instance();
        OP_REQUIRES_ASYNC(
            c, true == cluster->IsInitialized(),
            errors::InvalidArgument("cluster instance not initialized:"), done);

        std::vector<BnVarsPullCall*> calls;

        for (size_t shard_id = 0; shard_id < cluster->RankNum(); shard_id++) {
            calls.emplace_back(
                new BnVarsPullCall(table_handle_, shard_id));
        }

	BnVarsInfos bnVarsInfos(bn_size);

        Semaphore semaphore(calls.size());

        for (auto& call : calls) {
            call->Start([this, call, &bnVarsInfos, &semaphore]() {
				bnVarsInfos.Add(call->cntl.response_attachment());
                semaphore.Notify();
                delete call;
            });
        }

        semaphore.WaitForSemaphore();

	Eigen::ArrayXf weights_array = Eigen::ArrayXf::Constant(cluster->RankNum(), 1.0f);
	for (size_t i = 0; i < cluster->RankNum(); i++) {
            weights_array(i) = bnVarsInfos.batch_counts_[0](0);
        }

	float total_count = weights_array.sum();
	weights_array /= total_count;

        std::cout << "--------------------" << weights_array(0) << std::endl;

	Eigen::ArrayXf weighted_mean = Eigen::ArrayXf::Zero(bn_size);
	for (size_t i = 0; i < cluster->RankNum(); i++) {
            Eigen::ArrayXf moving_means_i = bnVarsInfos.moving_means_[i];
            weighted_mean += moving_means_i * weights_array[i];  
        }

	Eigen::ArrayXf weighted_var = Eigen::ArrayXf::Zero(bn_size);
        for (size_t i = 0; i < cluster->RankNum(); i++) {
            Eigen::ArrayXf moving_vars_i = bnVarsInfos.moving_vars_[i];
            weighted_var += moving_vars_i * weights_array[i];
        }

	auto& moving_mean_var = bn_vars[0];
	float* moving_mean_flat = moving_mean_var->tensor()->flat<float>().data();
	std::copy(weighted_mean.data(), weighted_mean.data() + weighted_mean.size(), moving_mean_flat);

        auto& moving_var_var = bn_vars[1];
        float* moving_var_flat = moving_var_var->tensor()->flat<float>().data();
        std::copy(weighted_var.data(), weighted_var.data() + weighted_var.size(), moving_var_flat);
        done();

        return;
    }

private:
    int table_handle_;
    int N_;
};

REGISTER_KERNEL_BUILDER(Name("BnVarsPull").Device(DEVICE_CPU),
                        BnVarsPullKernel);

class BnVarsSetKernel : public OpKernel {
public:
    explicit BnVarsSetKernel(OpKernelConstruction* c)
        : OpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
        OP_REQUIRES_OK(c, c->GetAttr("N", &N_));
    }

    void Compute(OpKernelContext* c) override {
        butil::IOBuf bn_vars_buf;
        std::cout << "N is " << N_;

        for (int i = 0; i < N_; i++) {
            const ResourceHandle &handle = HandleFromInput(c, i);

            Var *variable = nullptr;
            const auto status = LookupResource<Var, false>(c, handle, &variable);

            OP_REQUIRES_OK(c, status);
            CHECK(variable);
            Tensor *var_tensor = variable->tensor();
//            std::cout << "var_tensor: \n" << var_tensor->NumElements();
 //           float* data_ptr = var_tensor->flat<float>().data();
  //          int num_elements = var_tensor->flat<float>().size();
   //         for (int i = 0; i < num_elements; ++i) {  
    //           std::cout << data_ptr[i] << std::endl;  
     //        }

            bn_vars_buf.append_user_data(var_tensor->flat<float>().data(),
                                 var_tensor->NumElements() * sizeof(float),
                                  NoOpDeleter);
        }

        BnTable* table = BnTableRegistry::Instance()->Get(table_handle_);

        OP_REQUIRES(c, nullptr != table,
                errors::InvalidArgument("BnTable have not created yet, handle:",
                    table_handle_));

        OP_REQUIRES(c, 0 == table->Set(bn_vars_buf),
                errors::InvalidArgument("BnTable update vars failed"));

        table->PrintDetail();

        return;
    }

private:
    int table_handle_;
    int N_;
};


REGISTER_KERNEL_BUILDER(Name("BnVarsSet").Device(DEVICE_CPU),
                        BnVarsSetKernel);

};
