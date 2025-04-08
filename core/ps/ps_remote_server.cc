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

#include "core/ps/ps_remote_server.h"

#include <brpc/channel.h>
#include <brpc/server.h>
#include <butil/rand_util.h>

using namespace google::protobuf;

namespace tensornet {

namespace {

template <typename TypeRequest, typename TypeResponse>
class Call : public Closure {
public:
    explicit Call(const MethodDescriptor *method_dp,
                  std::shared_ptr<brpc::Channel> channel,
                  brpc::Controller *cntl,
                  const TypeRequest *req,
                  TypeResponse *resp,
                  Callback &&done,
                  int req_cnt = 1)
        : method_dp_(method_dp)
        , channel_(channel)
        , cntl_(cntl)
        , req_(req)
        , resp_(resp)
        , done_(done)
        , req_cnt_(req_cnt) {
        CHECK(nullptr != method_dp_);
        Process_();
    }

    void Run() {
        std::unique_ptr<Call> self_guard(this);
        if (cntl_->Failed()) {
            if (3 < req_cnt_) {
                LOG(ERROR) << method_dp_->name() << " retry fail";
                abort();
                // done_();
            } else {
                LOG(INFO) << method_dp_->name() << cntl_->ErrorText() << ", do retry[" << req_cnt_ << "]";

                // random sleep 1s~5s
                int sleep_us = butil::RandInt(1 * 1000 * 1000, 5 * 1000 * 1000);
                bthread_usleep(sleep_us);

                // backup request setting
                butil::IOBuf req_buf;
                auto http_method = cntl_->http_request().method();
                auto req_compress_type = cntl_->request_compress_type();
                cntl_->request_attachment().swap(req_buf);

                // reset error info
                cntl_->Reset();

                // repopulate
                cntl_->http_request().set_method(http_method);
                cntl_->set_request_compress_type(req_compress_type);
                cntl_->request_attachment().swap(req_buf);

                new Call<TypeRequest, TypeResponse>(method_dp_, channel_, cntl_, req_, resp_, std::move(done_),
                                                    req_cnt_ + 1);
            }

        } else {
            done_();
        }
    }

protected:
    void Process_() { channel_->CallMethod(method_dp_, cntl_, req_, resp_, this); }

private:
    const MethodDescriptor *method_dp_ = nullptr;

    std::shared_ptr<brpc::Channel> channel_;
    brpc::Controller *cntl_ = nullptr;
    const TypeRequest *req_ = nullptr;
    TypeResponse *resp_ = nullptr;
    Callback done_;
    int req_cnt_ = 0;
};

}  // namespace

PsRemoteServer::PsRemoteServer(std::shared_ptr<brpc::Channel> &channel)
    : channel_(channel) {
    sparse_pull_dp_ = PsService::descriptor()->FindMethodByName("SparsePull");
    sparse_push_dp_ = PsService::descriptor()->FindMethodByName("SparsePush");
    dense_push_pull_dp_ = PsService::descriptor()->FindMethodByName("DensePushPull");
    dataset_pull_dp_ = PsService::descriptor()->FindMethodByName("DatasetPull");
    bn_statistics_push_dp_ = PsService::descriptor()->FindMethodByName("BnStatisticsPush");
    bn_statistics_pull_dp_ = PsService::descriptor()->FindMethodByName("BnStatisticsPull");
}

PsRemoteServer::~PsRemoteServer() {}

void PsRemoteServer::SparsePullAsync(brpc::Controller *cntl,
                                     const SparsePullRequest *request,
                                     SparsePullResponse *response,
                                     Callback done) const {
    new Call<SparsePullRequest, SparsePullResponse>(sparse_pull_dp_, channel_, cntl, request, response,
                                                    std::move(done));
}

void PsRemoteServer::SparsePushAsync(brpc::Controller *cntl,
                                     const SparsePushRequest *request,
                                     SparsePushResponse *response,
                                     Callback done) const {
    new Call<SparsePushRequest, SparsePushResponse>(sparse_push_dp_, channel_, cntl, request, response,
                                                    std::move(done));
}

void PsRemoteServer::DensePushPullAsync(brpc::Controller *cntl,
                                        const DensePushPullRequest *request,
                                        DensePushPullResponse *response,
                                        Callback done) const {
    new Call<DensePushPullRequest, DensePushPullResponse>(dense_push_pull_dp_, channel_, cntl, request, response,
                                                          std::move(done));
}

void PsRemoteServer::DatasetPullAsync(brpc::Controller *cntl,
                                      const DatasetPullRequest *request,
                                      DatasetPullResponse *response,
                                      Callback done) const {
    new Call<DatasetPullRequest, DatasetPullResponse>(dataset_pull_dp_, channel_, cntl, request, response,
                                                      std::move(done));
}

void PsRemoteServer::BnStatisticsPushAsync(brpc::Controller *cntl,
                                           const BnStatisticsPushRequest *request,
                                           BnStatisticsPushResponse *response,
                                           Callback done) const {
    new Call<BnStatisticsPushRequest, BnStatisticsPushResponse>(bn_statistics_push_dp_, channel_, cntl, request,
                                                                response, std::move(done));
}

void PsRemoteServer::BnStatisticsPullAsync(brpc::Controller *cntl,
                                           const BnStatisticsPullRequest *request,
                                           BnStatisticsPullResponse *response,
                                           Callback done) const {
    new Call<BnStatisticsPullRequest, BnStatisticsPullResponse>(bn_statistics_pull_dp_, channel_, cntl, request,
                                                                response, std::move(done));
}

}  // namespace tensornet
