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

#include <iostream>

#include "core/ps/optimizer/data_struct.h"

namespace tensornet {

int const SERIALIZE_FMT_ID = std::ios_base::xalloc();

void SparseOptValue::Serialize(std::ostream& os, int dim) {
    switch (os.iword(SERIALIZE_FMT_ID)) {
        case SF_TXT:
            SerializeTxt_(os, dim);
            break;
        case SF_BIN:
            SerializeBin_(os, dim);
            break;
    }
}

void SparseOptValue::DeSerialize(std::istream& is, int dim) {
    switch (is.iword(SERIALIZE_FMT_ID)) {
        case SF_TXT:
            DeSerializeTxt_(is, dim);
            break;
        case SF_BIN:
            DeSerializeBin_(is, dim);
            break;
    }
}


} // namespace tensornet

