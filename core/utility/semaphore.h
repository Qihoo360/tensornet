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

#ifndef TENSORNET_UTILITY_SEMAPHORE_H_
#define TENSORNET_UTILITY_SEMAPHORE_H_

#include <assert.h>
#include <atomic>  // NOLINT
#include <chrono>  // NOLINT

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensornet {

class Semaphore {
public:
    Semaphore(int cnt)
        : notified_(cnt) {}
    ~Semaphore() {
        // In case the notification is being used to synchronize its own deletion,
        // force any prior notifier to leave its critical section before the object
        // is destroyed.
        tensorflow::mutex_lock l(mu_);
    }

    void Notify() {
        tensorflow::mutex_lock l(mu_);
        assert(!HasBeenNotified());
        notified_.fetch_sub(1);
        cv_.notify_one();
    }

    bool HasBeenNotified() const { return notified_.load(std::memory_order_acquire) == 0; }

    void WaitForSemaphore() {
        if (!HasBeenNotified()) {
            tensorflow::mutex_lock l(mu_);
            while (!HasBeenNotified()) {
                cv_.wait(l);
            }
        }
    }

private:
    friend bool WaitForSemaphoreWithTimeout(Semaphore* n, tensorflow::int64 timeout_in_us);
    bool WaitForSemaphoreWithTimeout(tensorflow::int64 timeout_in_us) {
        bool notified = HasBeenNotified();
        if (!notified) {
            tensorflow::mutex_lock l(mu_);
            do {
                notified = HasBeenNotified();
            } while (!notified && cv_.wait_for(l, std::chrono::microseconds(timeout_in_us)) != std::cv_status::timeout);
        }
        return notified;
    }

    tensorflow::mutex mu_;               // protects mutations of notified_
    tensorflow::condition_variable cv_;  // signaled when notified_ becomes non-zero
    std::atomic<int> notified_;          // mutations under mu_
};

inline bool WaitForSemaphoreWithTimeout(Semaphore* n, tensorflow::int64 timeout_in_us) {
    return n->WaitForSemaphoreWithTimeout(timeout_in_us);
}

}  // namespace tensornet

#endif  // TENSORNET_UTILITY_SEMAPHORE_H_
