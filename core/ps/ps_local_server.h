// Copyright 2020-2025 Qihoo Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORNET_PS_LOCAL_SERVER_H_
#define TENSORNET_PS_LOCAL_SERVER_H_

#include "core/ps/ps_server_interface.h"

namespace tensornet {

class PsLocalServer : public PsServerInterface {
public:
    PsLocalServer() {}

    ~PsLocalServer() {}

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
};

}  // namespace tensornet

#endif  // TENSORNET_PS_LOCAL_SERVER_H_
