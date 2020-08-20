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

#ifndef TENSORNET_UTILITY_MPI_MANAGER_H_
#define TENSORNET_UTILITY_MPI_MANAGER_H_

#include <vector>
#include <string>
#include <mpi.h>

#include <butil/logging.h>

namespace tensornet {

class MpiManager {
public:
    static MpiManager* Instance();

    int Init();

    int Rank() const {
        return rank_;
    }

    int RankNum() const {
        return rank_num_;
    }

    std::vector<std::string> GetWorkers();

    void Barrier();

private:
    MpiManager();
    ~MpiManager();

private:
    bool is_initialized_ = false;
    int rank_ = 0;
    int rank_num_ = 0;

    std::vector<std::string> ip_table_;
    std::vector<uint16_t> port_table_;
};

} // namespace tensornet

#endif // TENSORNET_UTILITY_MPI_MANAGER_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
