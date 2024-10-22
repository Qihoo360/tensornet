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

#include "core/ps/ps_local_server.h"
#include "core/ps/ps_cluster.h"
#include "core/ps_interface/ps_server.pb.h"
#include "core/ps/table/dense_table.h"
#include "core/ps/table/sparse_table.h"
#include "core/ps/table/bn_table.h"
#include "core/kernels/data/balance_dataset_ops.h"
#include "core/ps/optimizer/optimizer_kernel.h"

#include <brpc/server.h>

namespace tensornet {

void PsLocalServer::SparsePullAsync(brpc::Controller *cntl,
                                    const SparsePullRequest *request,
                                    SparsePullResponse *response,
                                    Callback done) const {
    SparseTable *table =
        SparseTableRegistry::Instance()->Get(request->table_handle());
    CHECK(nullptr != table);

    butil::IOBuf& output = cntl->response_attachment();
    table->Pull(request, output, response);

    done();
}

void PsLocalServer::SparsePushAsync(brpc::Controller *cntl,
                                    const SparsePushRequest *request,
                                    SparsePushResponse *response,
                                    Callback done) const {
    SparseTable *table =
        SparseTableRegistry::Instance()->Get(request->table_handle());
    CHECK(nullptr != table);

    butil::IOBuf& grad_buf = cntl->request_attachment();
    table->Push(request, grad_buf, response);

    done();
}

void PsLocalServer::DensePushPullAsync(brpc::Controller *cntl,
                                       const DensePushPullRequest *request,
                                       DensePushPullResponse *response,
                                       Callback done) const {
    DenseTable *table =
        DenseTableRegistry::Instance()->Get(request->table_handle());
    CHECK(nullptr != table);

    int shard_id = PsCluster::Instance()->Rank();

    const auto opt_kernel = table->GetOptKernels(shard_id);

    CHECK(nullptr != opt_kernel);

    butil::IOBuf& grad_buf = cntl->request_attachment();
    opt_kernel->Apply(grad_buf);

    butil::IOBuf& output = cntl->response_attachment();
    opt_kernel->GetWeight(output);

    done();
}

void PsLocalServer::DatasetPullAsync(brpc::Controller *cntl,
                                     const DatasetPullRequest *request,
                                     DatasetPullResponse *response,
                                     Callback done) const {
    tensorflow::BalanceInputDataInfo::Instance()
        ->ProcessBrpcDatasetPullReq(request, response);

    done();
}

void PsLocalServer::BnStatisticsPushAsync(brpc::Controller *cntl,
                                     const BnStatisticsPushRequest *request,
                                     BnStatisticsPushResponse *response,
                                     Callback done) const {
    BnTable *table = BnTableRegistry::Instance()->Get(request->table_handle());
	CHECK(nullptr != table);
	butil::IOBuf& acc_data = cntl->request_attachment();
	table->Append(acc_data, false);

    done();
}

void PsLocalServer::BnStatisticsPullAsync(brpc::Controller *cntl,
                                     const BnStatisticsPullRequest *request,
                                     BnStatisticsPullResponse *response,
                                     Callback done) const {
    BnTable *table = BnTableRegistry::Instance()->Get(request->table_handle());
        CHECK(nullptr != table);
        response->set_table_handle(request->table_handle());
        butil::IOBuf& bn_statistics_buf = cntl->response_attachment();
        table->GetIncStatistics(bn_statistics_buf);

    done();
}
}  // namespace tensornet
