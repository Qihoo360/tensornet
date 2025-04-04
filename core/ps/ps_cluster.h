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

#ifndef TENSORNET_PS_CLUSTER_H_
#define TENSORNET_PS_CLUSTER_H_

#include <string>
#include <map>
#include <memory>

#include "core/ps/ps_service_impl.h"
#include "core/ps/ps_local_server.h"
#include "core/ps/ps_remote_server.h"

namespace brpc {
// NOTE! do not include brpc/server.h in this header file,
// its will cause butil LOG macro conflict with tensorflow LOG macro
class Server;

} // namespace brpc {

namespace tensornet {

class PsCluster {
public:
    static PsCluster* Instance();

    int Init();

    bool IsInitialized() const {
        return is_initialized_;
    }

    size_t RankNum() const;

    int Rank() const;

    const PsServerInterface* GetServer(int shard_id) const;

    void Barrier() const;

public:
    PsLocalServer local_server;

private:
    PsCluster();

    ~PsCluster();

    int InitRemoteServers_();

    uint16_t GetSelfPort_();

private:
    bool is_initialized_ = false;
    std::unique_ptr<brpc::Server> server_;

    PsServiceImpl ps_service_impl_;

    std::vector<std::unique_ptr<PsRemoteServer>> remote_servers_;

    std::vector<std::string> workers_;
};

} // namespace tensornet

#endif  // TENSORNET_PS_CLUSTER_H_
