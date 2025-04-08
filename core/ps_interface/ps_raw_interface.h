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

#ifndef TENSORNET_PS_INTERFACE_RAW_INTERFACE_H_
#define TENSORNET_PS_INTERFACE_RAW_INTERFACE_H_

#include <stdint.h>

namespace tensornet {

struct SparsePushSignInfo {
public:
    SparsePushSignInfo()
        : SparsePushSignInfo(0, 0, 0) {}

    SparsePushSignInfo(uint64_t s, int bs, int cs)
        : sign(s)
        , batch_show(bs)
        , batch_click(cs) {}

    uint64_t sign;
    int batch_show;
    int batch_click;
};

}  // namespace tensornet

#endif  // TENSORNET_PS_INTERFACE_RAW_INTERFACE_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
