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

#ifndef TENSORNET_KERNEL_RESOURCE_VAR_WRAPPER_H_
#define TENSORNET_KERNEL_RESOURCE_VAR_WRAPPER_H_

// trick: do not include resource_mgr.h, use symbol define at _pywrap_tensorflow_internal.so instead.
// because of TypeIndex::Make<T>(name) function use static variable in core/framework/type_index.h,
// they will generate different hash code for the same Symbol in different so file.
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_

// copy from tensorflow source code, those code must be same as its define in resource_mgr.h
namespace tensorflow {
class ResourceBase : public core::RefCounted {
public:
    // Returns a debug string for *this.
    virtual string DebugString() const = 0;

    // Returns memory used by this resource.
    virtual int64 MemoryUsed() const { return 0; }
};
} // namespace tensorflow

#include "tensorflow/core/framework/resource_var.h"

#undef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_

#endif // !TENSORNET_KERNEL_RESOURCE_VAR_WRAPPER_H_
