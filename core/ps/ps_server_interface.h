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

#ifndef TENSORNET_PS_SERVER_INTERFACE_H_
#define TENSORNET_PS_SERVER_INTERFACE_H_

#include <functional>

#include "core/ps_interface/ps_server.pb.h"

namespace brpc {

class Controller;

}  // namespace brpc

namespace tensornet {

typedef std::function<void()> Callback;

class PsServerInterface {
public:
    virtual ~PsServerInterface() {};

    virtual void SparsePullAsync(brpc::Controller *cntl,
                                 const SparsePullRequest *request,
                                 SparsePullResponse *response,
                                 Callback done) const = 0;

    virtual void SparsePushAsync(brpc::Controller *cntl,
                                 const SparsePushRequest *request,
                                 SparsePushResponse *response,
                                 Callback done) const = 0;

    virtual void DensePushPullAsync(brpc::Controller *cntl,
                                    const DensePushPullRequest *request,
                                    DensePushPullResponse *response,
                                    Callback done) const = 0;

    virtual void DatasetPullAsync(brpc::Controller *cntl,
                                  const DatasetPullRequest *request,
                                  DatasetPullResponse *response,
                                  Callback done) const = 0;

    virtual void BnStatisticsPushAsync(brpc::Controller *cntl,
                                 const BnStatisticsPushRequest *request,
                                 BnStatisticsPushResponse *response,
                                 Callback done) const = 0;

    virtual void BnStatisticsPullAsync(brpc::Controller *cntl,
                                 const BnStatisticsPullRequest *request,
                                 BnStatisticsPullResponse *response,
                                 Callback done) const = 0;

private:
    typedef PsServerInterface ME;
};

}  // namespace tensornet

#endif  // TENSORNET_PS_SERVER_INTERFACE_H_
