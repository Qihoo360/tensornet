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

#ifndef TENSORNET_CORE_PUBLIC_VERSION_H_
#define TENSORNET_CORE_PUBLIC_VERSION_H_

#include "tensorflow/core/public/version.h"

// tensornet support tensorflow-2.3, the following macro is compatible with tensorflow-2.2
// https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0
#if (TF_MAJOR_VERSION == 2) && (TF_MINOR_VERSION == 2) 
    #define TN_COMPATIBLE_INTERFACE_2_2 1
#endif

#endif  // TENSORNET_CORE_PUBLIC_VERSION_H_
