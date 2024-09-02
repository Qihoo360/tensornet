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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("UpdateMoments")
    .Doc(R"doc(pull mean, var, count from parameter server)doc")
    .Input("vars: N * resource")
    .Attr("table_handle: int")
    .Attr("N: int")
	.SetShapeFn(shape_inference::NoOutputs);


REGISTER_OP("BnStatisticsPush")
    .Doc(R"doc(save local bn vars to ps)doc")
    .Input("vars: N * resource")
    .Attr("table_handle: int")
    .Attr("N: int")
    .Attr("synchronized: bool")
    .SetShapeFn(shape_inference::NoOutputs);


REGISTER_OP("BnStatisticsPull")
    .Doc(R"doc(save local bn vars to ps)doc")
    .Input("vars: N * resource")
    .Attr("table_handle: int")
    .Attr("N: int")
    .SetShapeFn(shape_inference::NoOutputs);
