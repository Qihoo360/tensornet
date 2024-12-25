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

#ifndef TENSORNET_OPTIMIZER_DATA_STRUCT_H_
#define TENSORNET_OPTIMIZER_DATA_STRUCT_H_

namespace tensornet {

struct SparseGradInfo {
    float* grad;
    int batch_show;
    int batch_click;
};

extern int const SERIALIZE_FMT_ID;

enum SerializeFormat {
    SF_TXT,
    SF_BIN,
};

class alignas(4) SparseOptValue {
public:
    void ShowDecay(float decay_rate) {
        show_ = (1 - decay_rate) * delta_show_ + decay_rate * show_;
        delta_show_ = 0;
    }

    void Serialize(std::ostream& os, int dim);

    void DeSerialize(std::istream& is, int dim);

    void SetOldCompat(bool old_compat) {
	old_compat_ = old_compat;
    }

    float Show() const {
        return show_;
    }

protected:
    virtual void SerializeTxt_(std::ostream& os, int dim) = 0;
    virtual void DeSerializeTxt_(std::istream& is, int dim) = 0;
    virtual void SerializeBin_(std::ostream& os, int dim) = 0;
    virtual void DeSerializeBin_(std::istream& is, int dim) = 0;

protected:
    float show_ = 0.0;
    float click_ = 0.0;
    int delta_show_ = 0;
    bool old_compat_ = false;
};

} // namespace tensornet {

#endif // !TENSORNET_OPTIMIZER_DATA_STRUCT_H_
