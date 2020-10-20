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

#include "core/ps/ps_cluster.h"
#include "core/utility/mpi_manager.h"

#include <brpc/server.h>
#include <brpc/channel.h>

namespace tensornet {

PsCluster::PsCluster() {
    server_ = std::make_unique<brpc::Server>();
}

PsCluster::~PsCluster() {
}

PsCluster* PsCluster::Instance() {
    static PsCluster cluster;

    return &cluster;
}

int PsCluster::Init() {
    if (is_initialized_) {
        return -1;
    }

    if (server_->AddService(&ps_service_impl_, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "Fail to add ps_service_impl";
        return -1;
    }

    MpiManager* mpi_manager = MpiManager::Instance();

    CHECK_EQ(0, mpi_manager->Init());

    workers_ = mpi_manager->GetWorkers();

    CHECK_GT(workers_.size(), 0);

    CHECK_EQ(0, InitRemoteServers_());

    uint16_t self_port = GetSelfPort_();

    brpc::ServerOptions server_options;

    if (server_->Start(self_port, &server_options) != 0) {
        LOG(ERROR) << "tensornet fail to bind port:" << self_port;
        return -1;
    }

    Barrier();

    is_initialized_ = true;

    return 0;
}

size_t PsCluster::RankNum() const {
    return MpiManager::Instance()->RankNum();
}

int PsCluster::Rank() const {
    return MpiManager::Instance()->Rank();
}

int PsCluster::InitRemoteServers_() {
    brpc::ChannelOptions options;

    options.protocol = "baidu_std";
    options.connection_type = "single";
    options.timeout_ms = 60000;
    options.max_retry = 1;

    for (size_t i = 0; i < workers_.size(); i++) {
        std::shared_ptr<brpc::Channel> channel =
            std::make_shared<brpc::Channel>();

        if (channel->Init(workers_[i].c_str(), "", &options) != 0) {
            LOG(ERROR) << "Fail to initialize channel with " << workers_[i];
            return -1;
        }

        remote_servers_.emplace_back(std::make_unique<PsRemoteServer>(channel));
    }

    return 0;
}

const PsServerInterface* PsCluster::GetServer(int shard_id) const {
    if (Rank() == shard_id) {
        return &local_server;
    } else {
        CHECK_LT(shard_id, (int)remote_servers_.size());
        return remote_servers_[shard_id].get();
    }
}

void PsCluster::Barrier() const {
    MpiManager::Instance()->Barrier();
}

uint16_t PsCluster::GetSelfPort_() {
    const std::string& worker = workers_[Rank()];

    int pos = worker.find(':') + 1;

    CHECK_LT(pos, worker.size());

    std::string port_str = worker.substr(pos);

    return std::stoul(port_str);
}

} // namespace tensornet
