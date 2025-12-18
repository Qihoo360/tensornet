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

#include "tensorflow/core/public/version.h"

#ifndef TENSORNET_KERNEL_RESOURCE_VAR_WRAPPER_H_
#define TENSORNET_KERNEL_RESOURCE_VAR_WRAPPER_H_

// trick: do not include resource_mgr.h, use symbols (tensorflow::Var::*, LookupResource<Var, false>(...)) define at
// _pywrap_tensorflow_internal.so instead. because of TypeIndex::Make<T>(name) function use static variable in
// core/framework/type_index.h, they will generate different hash code for the same Symbol in different so file.
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_

// 2.7: https://github.com/tensorflow/tensorflow/commit/69fc036bb636ea7f76d4b4d879e27f7e4cf0ad33
// 2.8: https://github.com/tensorflow/tensorflow/commit/efd91a1709b033c50210649a008b71f3882ce3b5

#if (TF_MAJOR_VERSION > 2) || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 8)

#elif (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION == 7)
// TF == 2.7: include resource_base.h and use current implementation
#include "tensorflow/core/framework/resource_base.h"

#else
// TF < 2.7: use current implementation
// copy from tensorflow source code, those code must be same as its define in resource_mgr.h
namespace tensorflow {
class ResourceBase : public core::RefCounted {
public:
    // Returns a debug string for *this.
    virtual string DebugString() const = 0;

    // Returns memory used by this resource.
    virtual int64 MemoryUsed() const { return 0; }
};
}  // namespace tensorflow

#endif

#include "tensorflow/core/framework/resource_var.h"

#undef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_

#endif  // !TENSORNET_KERNEL_RESOURCE_VAR_WRAPPER_H_
