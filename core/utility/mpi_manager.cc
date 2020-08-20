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

#include "core/utility/mpi_manager.h"
#include "core/utility/net_util.h"

namespace tensornet {

#define MPICHECK(cmd) do {                              \
    int e = cmd;                                        \
    if( e != MPI_SUCCESS ) {                            \
        printf("Failed: MPI error %s:%d '%d'\n",        \
                __FILE__,__LINE__, e);                  \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)


MpiManager::MpiManager() { }

MpiManager::~MpiManager() {
    if (is_initialized_) {
        MPICHECK(MPI_Finalize());
    }
}

MpiManager* MpiManager::Instance() {
    static MpiManager instance;

    return &instance;
}

int MpiManager::Init() {
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &rank_num_));

    ip_table_.resize(rank_num_);
    port_table_.resize(rank_num_);

    ip_table_[rank_] = get_local_ip_internal();
    port_table_[rank_] = get_useable_port();

    for (int rank = 0; rank < rank_num_; ++rank) {
        int len = ip_table_[rank].size();
        MPI_Bcast(&len, 1, MPI_INT, rank, MPI_COMM_WORLD);

        if (Rank() != rank) {
            ip_table_[rank].resize(len);
        }

        MPI_Bcast(const_cast<char*>(ip_table_[rank].data()), len,
                MPI_BYTE, rank, MPI_COMM_WORLD);
    }

    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_SHORT, &port_table_[0], 1,
                             MPI_SHORT, MPI_COMM_WORLD));

    is_initialized_ = true;

    return 0;
}

void MpiManager::Barrier() {
    std::vector<MPI_Request> reqs(RankNum(), MPI_REQUEST_NULL);
    int dummy = 0;

    for (int i = 0; i < RankNum(); ++i) {
        MPI_Irecv(&dummy, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &reqs[i]);
    }

    for (int i = 0; i < RankNum(); ++i) {
        MPI_Send(&dummy, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < RankNum(); ++i) {
        for (unsigned long x = 1;; x = std::min(x * 2, 2000UL)) {
            int flag = 0;
            MPI_Test(&reqs[i], &flag, MPI_STATUSES_IGNORE);
            if (flag) {
                break;
            }
            usleep(x);
        }
    }
}

std::vector<std::string> MpiManager::GetWorkers() {
    std::vector<std::string> workers;

    for (size_t i = 0; i < ip_table_.size(); i++) {
        std::string worker = ip_table_[i] + ":" + std::to_string(port_table_[i]);
        workers.push_back(worker);
    }

    return workers;
}

} // namespace tensornet

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
