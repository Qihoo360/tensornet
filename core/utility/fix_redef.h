// Copyright (c) 2025, Qihoo, Inc.  All rights reserved.
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

#ifndef TENSORNET_UTILITY_FIX_REDEF_H_
#define TENSORNET_UTILITY_FIX_REDEF_H_

#include <brpc/traceprintf.h>
#include <butil/logging.h>

#undef TRACEPRINTF

#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT

#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LE
#undef DCHECK_LT
#undef DCHECK_GE
#undef DCHECK_GT

#undef VLOG_IS_ON
#undef VLOG
#undef LOG
#undef DVLOG

#undef LOG_EVERY_N
#undef LOG_FIRST_N

#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/tracing.h>

#endif  // TENSORNET_UTILITY_FIX_REDEF_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
