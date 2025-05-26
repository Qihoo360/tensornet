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

#ifndef TENSORNET_UTILITY_NET_UTIL_H_
#define TENSORNET_UTILITY_NET_UTIL_H_

#include <stdint.h>
#include <string>

namespace tensornet {

uint16_t get_useable_port();

extern std::string get_local_ip_internal();

}  // namespace tensornet

#endif  // __COMMON_NET_UTIL_H__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
