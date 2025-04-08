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

#ifndef TENSORNET_UTILITY_ALLOCATOR_H_
#define TENSORNET_UTILITY_ALLOCATOR_H_

#include <atomic>
#include <ctime>
#include <random>

#include <butil/logging.h>

namespace tensornet {

template <typename T>
class Allocator {
public:
    Allocator(int type_sizeof)
        : Allocator(type_sizeof, 1 << 16) {}

    Allocator(int type_sizeof, int block_len)
        : pool_(nullptr)
        , type_sizeof_(type_sizeof)
        , block_len_(block_len)
        , useable_block_(nullptr) {
        CHECK_GE(type_sizeof, sizeof(T));
        CHECK_GE(type_sizeof, sizeof(Block));

        create_new_pool_();
    }

    ~Allocator() {
        while (pool_) {
            Pool* next = pool_->next;
            free(pool_);
            pool_ = next;
        }
    }

    Allocator(Allocator&& other)
        : pool_(other.pool_)
        , type_sizeof_(other.type_sizeof_)
        , block_len_(other.block_len_)
        , useable_block_(other.useable_block_) {
        other.pool_ = nullptr;
        other.useable_block_ = nullptr;
    }

    Allocator operator=(Allocator&& other) {
        pool_ = other.pool_;
        type_sizeof_ = other.type_sizeof_;
        block_len_ = other.block_len_;
        useable_block_ = other.useable_block_;

        other.pool_ = nullptr;
        other.useable_block_ = nullptr;
    }

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    template <class... ARGS>
    T* allocate(ARGS&&... args) {
        if (!useable_block_) {
            create_new_pool_();
        }

        T* value = (T*)useable_block_;
        useable_block_ = useable_block_->next;

        new (value) T(std::forward<ARGS>(args)...);

        return value;
    }

    void deallocate(T* ptr) {
        ptr->~T();
        auto block = (Block*)ptr;
        block->next = useable_block_;
        useable_block_ = block;
    }

private:
    void create_new_pool_() {
        Pool* pool = nullptr;

        PCHECK(0 == posix_memalign((void**)&pool, alignof(Pool), sizeof(Pool) + type_sizeof_ * block_len_));

        pool->next = pool_;
        pool_ = pool;

        for (int i = 0; i < block_len_; i++) {
            auto block = (Block*)(pool_->block + type_sizeof_ * i);
            block->next = useable_block_;
            useable_block_ = block;
        }

        return;
    }

private:
    union Block {
        Block* next;
        char data[0];
    };

    struct Pool {
        Pool* next;
        char block[0];
    };

    Pool* pool_ = nullptr;

    int type_sizeof_ = 0;
    int block_len_ = 0;

    Block* useable_block_ = nullptr;
};

}  // namespace tensornet

#endif  // TENSORNET_UTILITY_ALLOCATOR_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
