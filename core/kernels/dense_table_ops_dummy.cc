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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class DenseTableInitKernel : public OpKernel {
public:
    explicit DenseTableInitKernel(OpKernelConstruction* c)
        : OpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
    }

    void Compute(OpKernelContext* c) override { return; }

private:
    int table_handle_;
};

REGISTER_KERNEL_BUILDER(Name("DenseTableInit").Device(DEVICE_CPU), DenseTableInitKernel);

class DenseTablePushPullKernel : public AsyncOpKernel {
public:
    explicit DenseTablePushPullKernel(OpKernelConstruction* c)
        : AsyncOpKernel(c) {
        OP_REQUIRES_OK(c, c->GetAttr("table_handle", &table_handle_));
    }

    void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
        done();
        return;
    }

private:
    int table_handle_;
};

REGISTER_KERNEL_BUILDER(Name("DenseTablePushPull").Device(DEVICE_CPU), DenseTablePushPullKernel);

}  // namespace tensorflow
