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

#ifndef TENSORNET_PS_SERVICE_IMPL_H_
#define TENSORNET_PS_SERVICE_IMPL_H_

#include "core/protobuf/ps_server.pb.h"

namespace tensornet {

class PsServiceImpl : public PsService {
public:
    PsServiceImpl();
    virtual ~PsServiceImpl();

    virtual void SparsePull(google::protobuf::RpcController* cntl_base,
                            const SparsePullRequest* request,
                            SparsePullResponse* response,
                            google::protobuf::Closure* done);

    virtual void SparsePush(google::protobuf::RpcController* cntl_base,
                            const SparsePushRequest* request,
                            SparsePushResponse* response,
                            google::protobuf::Closure* done);

    virtual void DensePushPull(google::protobuf::RpcController* cntl_base,
                               const DensePushPullRequest* request,
                               DensePushPullResponse* response,
                               google::protobuf::Closure* done);

    virtual void DatasetPull(google::protobuf::RpcController* cntl_base,
                             const DatasetPullRequest* request,
                             DatasetPullResponse* response,
                             google::protobuf::Closure* done);
};

}  // end of namespace tensornet

#endif  // TENSORNET_PS_SERVICE_IMPL_H_
