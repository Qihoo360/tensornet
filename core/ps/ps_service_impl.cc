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

#include "core/ps/ps_service_impl.h"

#include <brpc/server.h>

#include "core/ps/ps_cluster.h"

namespace tensornet {

PsServiceImpl::PsServiceImpl() {}
PsServiceImpl::~PsServiceImpl() {}

void PsServiceImpl::SparsePull(google::protobuf::RpcController* cntl_base,
                               const SparsePullRequest* request,
                               SparsePullResponse* response,
                               google::protobuf::Closure* done) {
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
    cntl->set_response_compress_type(brpc::COMPRESS_TYPE_GZIP);

    auto* cluster = PsCluster::Instance();
    const auto* si = cluster->GetServer(cluster->Rank());

    si->SparsePullAsync(cntl, request, response,
                        [done]() { done->Run(); });
}

void PsServiceImpl::SparsePush(google::protobuf::RpcController* cntl_base,
                               const SparsePushRequest* request,
                               SparsePushResponse* response,
                               google::protobuf::Closure* done) {
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
    cntl->set_response_compress_type(brpc::COMPRESS_TYPE_GZIP);

    auto* cluster = PsCluster::Instance();
    const auto* si = cluster->GetServer(cluster->Rank());

    si->SparsePushAsync(cntl, request, response,
                        [done]() { done->Run(); });
}

void PsServiceImpl::DensePushPull(google::protobuf::RpcController* cntl_base,
                                  const DensePushPullRequest* request,
                                  DensePushPullResponse* response,
                                  google::protobuf::Closure* done) {
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
    cntl->set_response_compress_type(brpc::COMPRESS_TYPE_GZIP);

    auto* cluster = PsCluster::Instance();
    const auto* si = cluster->GetServer(cluster->Rank());

    si->DensePushPullAsync(cntl, request, response,
                           [done]() { done->Run(); });
}

void PsServiceImpl::DatasetPull(google::protobuf::RpcController* cntl_base,
                                const DatasetPullRequest* request,
                                DatasetPullResponse* response,
                                google::protobuf::Closure* done) {
    brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
    cntl->set_response_compress_type(brpc::COMPRESS_TYPE_GZIP);

    auto* cluster = PsCluster::Instance();
    const auto* si = cluster->GetServer(cluster->Rank());

    si->DatasetPullAsync(cntl, request, response,
                         [done]() { done->Run(); });
}

}  // end of namespace tensornet
