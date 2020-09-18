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

REGISTER_OP("SparseTablePull")
    .Doc(R"doc(pull variable from parameter server
    )doc")
    .Input("resources: N * resource")
    .Input("values: N * int64")
    .Output("mapped_values: N * int64")
    .Attr("table_handle: int")
    .Attr("N: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int N = 0;

        TF_CHECK_OK(c->GetAttr("N", &N));

        for (int i = 0; i < N; i++) {
            shape_inference::ShapeHandle shape;

            TF_RETURN_IF_ERROR(c->WithRank(c->input(N + i), 1, &shape));

            c->set_output(i, shape);
        }

        return Status::OK();
    });


REGISTER_OP("SparseTablePush")
    .Doc(R"doc(push variable from parameter server
    )doc")
    .Input("values: N * int64")
    .Input("grads: N * float")
    .Attr("table_handle: int")
    .Attr("N: int")
    .SetShapeFn(shape_inference::NoOutputs);
