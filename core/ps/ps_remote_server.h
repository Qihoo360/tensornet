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

#ifndef TENSORNET_PS_REMOTE_SERVER_H_
#define TENSORNET_PS_REMOTE_SERVER_H_

#include "core/ps/ps_server_interface.h"

namespace brpc {

class Channel;

}  // namespace brpc

namespace tensornet {

class PsRemoteServer : public PsServerInterface {
public:
    PsRemoteServer(std::shared_ptr<brpc::Channel> &channel);

    ~PsRemoteServer();

    virtual void SparsePullAsync(brpc::Controller *cntl,
                                 const SparsePullRequest *request,
                                 SparsePullResponse *response,
                                 Callback done) const override;

    virtual void SparsePushAsync(brpc::Controller *cntl,
                                 const SparsePushRequest *request,
                                 SparsePushResponse *response,
                                 Callback done) const override;

    virtual void DensePushPullAsync(brpc::Controller *cntl,
                                    const DensePushPullRequest *request,
                                    DensePushPullResponse *response,
                                    Callback done) const override;

    virtual void DatasetPullAsync(brpc::Controller *cntl,
                                  const DatasetPullRequest *request,
                                  DatasetPullResponse *response,
                                  Callback done) const override;

    virtual void BnStatisticsPushAsync(brpc::Controller *cntl,
                                 const BnStatisticsPushRequest *request,
                                 BnStatisticsPushResponse *response,
                                 Callback done) const override;

    virtual void BnStatisticsPullAsync(brpc::Controller *cntl,
                                 const BnStatisticsPullRequest *request,
                                 BnStatisticsPullResponse *response,
                                 Callback done) const override;

private:
    std::shared_ptr<brpc::Channel> channel_;

    const google::protobuf::MethodDescriptor* sparse_pull_dp_ = nullptr;
    const google::protobuf::MethodDescriptor* sparse_push_dp_ = nullptr;
    const google::protobuf::MethodDescriptor* dense_push_pull_dp_ = nullptr;
    const google::protobuf::MethodDescriptor* dataset_pull_dp_ = nullptr;
    const google::protobuf::MethodDescriptor* bn_statistics_push_dp_ = nullptr;
    const google::protobuf::MethodDescriptor* bn_statistics_pull_dp_ = nullptr;
};

}  // namespace tensornet

#endif  // TENSORNET_PS_REMOTE_SERVER_H_
