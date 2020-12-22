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

#ifndef TENSORNET_OPTIMIZER_KERNEL_H_
#define TENSORNET_OPTIMIZER_KERNEL_H_

#include "core/ps/optimizer/optimizer.h"

#include <mutex>
#include <unordered_map>
#include <functional>
#include <thread>
#include <algorithm>

#include <butil/iobuf.h>
#include <butil/logging.h>
#include <Eigen/Dense>

#include <boost/iostreams/stream.hpp>

#include "core/utility/file_io.h"
#include "core/utility/allocator.h"

#include "core/ps/optimizer/data_struct.h"

namespace tensornet {

static constexpr size_t DENSE_KERNEL_BLOCK_NUM = 8;
static constexpr size_t SPARSE_KERNEL_BLOCK_NUM = 8;

class DenseOptimizerKernelBase {
public:
    DenseOptimizerKernelBase(int off_b, int off_e)
        : off_b_(off_b)
        , off_e_(off_e) {
        assert(off_e > off_b);
    }

    virtual ~DenseOptimizerKernelBase() { }

    int OffsetBegin() const {
        return off_b_;
    }

    int OffsetEnd() const {
        return off_e_;
    }

    size_t Length() const {
        return size_t(off_e_ - off_b_);
    }

    virtual void Apply(butil::IOBuf& grad) = 0;

    virtual void SetWeight(butil::IOBuf& w_buf) = 0;

    virtual void GetWeight(butil::IOBuf& w_buf) const = 0;

    virtual void Serialized(std::ostream& os) const = 0;

    virtual void DeSerialized(std::istream& is) = 0;

    virtual size_t DataSize() = 0;

private:
    int off_b_ = 0;
    int off_e_ = 0;
};

class SparseOptimizerKernelBase {
public:
    SparseOptimizerKernelBase() {}

    ~SparseOptimizerKernelBase() {}

    virtual float* GetWeight(uint64_t sign) = 0;

    virtual void Apply(uint64_t sign, SparseGradInfo& grad_info) = 0;

    virtual void Serialized(const std::string& filepath) = 0;

    virtual void DeSerialized(const std::string& filepath) = 0;

    virtual size_t KeyCount() const = 0;

    virtual void ShowDecay() = 0;
};

template <typename OptType, typename ValueType>
class DenseKernelBlock {
public:
    DenseKernelBlock(size_t block_size, const OptimizerBase* opt)
        : block_size_(block_size)
        , opt_(dynamic_cast<const OptType*>(opt))
        , value_(opt_, block_size){
        mu_ = std::make_unique<std::mutex>();
    }

    DenseKernelBlock(DenseKernelBlock&) = delete;
    DenseKernelBlock& operator=(DenseKernelBlock&) = delete;

    DenseKernelBlock(DenseKernelBlock&& other)
        : mu_(std::move(other.mu_))
        , block_size_(other.block_size_)
        , opt_ (other.opt_)
        , value_(std::move(other.value_))
    { }

    DenseKernelBlock& operator=(DenseKernelBlock&& other) {
        mu_ = std::move(other.mu_);
        block_size_ = other.block_size_;
        opt_ = other.opt_;
        value_ = std::move(other.value_);
    }

    size_t BlockSize() const {
        return block_size_;
    }

    void SetWeight(butil::IOBuf& w_buf) {
        const std::lock_guard<std::mutex> lock(*mu_);
        value_.SetWeight(w_buf);
    }

    const Eigen::ArrayXf& GetWeight() const {
        return value_.GetWeight();
    }

    void Apply(const Eigen::ArrayXf& g) {
        const std::lock_guard<std::mutex> lock(*mu_);
        value_.Apply(opt_, g);
    }

    size_t DataSize() const {
        return value_.DataSize();
    }

    friend std::ostream& operator<<(std::ostream& os, const DenseKernelBlock& block) {
        const std::lock_guard<std::mutex> lock(*block.mu_);

        os << "opt_name:" << block.opt_->Name() << std::endl;
        os << block.value_ << std::endl;

        return os;
    }

    friend std::istream& operator>>(std::istream& is, DenseKernelBlock& block) {
        const std::lock_guard<std::mutex> lock(*block.mu_);

        std::string name;
        is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> name;

        CHECK_EQ(name, block.opt_->Name()) << "last trained model with optimizer is:" << name
            << " but current model use:" << block.opt_->Name() << " instead."
            << " you must make sure that use same optimizer when incremental training";

        is >> block.value_;

        return is;
    }

private:
    std::unique_ptr<std::mutex> mu_;
    size_t block_size_ = 0;
    const OptType* opt_ = nullptr;

    ValueType value_;
};

template <typename KernelBlockType>
class DenseOptimizerKernel : public DenseOptimizerKernelBase {
public:
    DenseOptimizerKernel(const OptimizerBase* opt, int off_b, int off_e)
        : DenseOptimizerKernelBase(off_b, off_e) {
        size_t per_block_size = std::ceil(Length() * 1.0 / DENSE_KERNEL_BLOCK_NUM);

        // make block and execute in parallel, every block has one lock
        for (size_t block_offset = 0; block_offset < Length();
                block_offset += per_block_size) {
            size_t block_size = per_block_size;

            if (block_offset + per_block_size > Length()) {
                block_size = Length() - block_offset;
            }

            blocks_.emplace_back(block_size, opt);
        }
    }

    virtual ~DenseOptimizerKernel() { }

    virtual void Apply(butil::IOBuf& grad) {
        assert(grad.size() == Length() * sizeof(float));

        for (size_t i = 0; i < blocks_.size(); i++) {
            size_t block_size = blocks_[i].BlockSize();

            Eigen::ArrayXf g(block_size);

            CHECK_EQ(sizeof(float) * block_size, grad.cutn(g.data(), sizeof(float) * block_size));

            blocks_[i].Apply(g);
        }
    }

    virtual void SetWeight(butil::IOBuf& w_buf) {
        for (size_t i = 0; i < blocks_.size(); i++) {
            butil::IOBuf buf;
            int length = blocks_[i].BlockSize() * sizeof(float);
            CHECK_EQ(length , w_buf.cutn(&buf, length));

            blocks_[i].SetWeight(buf);
        }
    }

    virtual void GetWeight(butil::IOBuf& w_buf) const {
        for (size_t i = 0; i < blocks_.size(); i++) {
            const auto& w = blocks_[i].GetWeight();
            w_buf.append(w.data(), w.size() * sizeof(float));
        }
    }

    virtual void Serialized(std::ostream& os) const {
        for (size_t i = 0; i < blocks_.size(); i++) {
            os << blocks_[i] << std::endl;
        }
        return;
    }

    virtual void DeSerialized(std::istream& is) {
        for (size_t i = 0; i < blocks_.size(); i++) {
            is >> blocks_[i];
        }
    }

    virtual size_t DataSize() {
        size_t total_size = 0;
        for (size_t i = 0; i < blocks_.size(); i++) {
            total_size += blocks_[i].DataSize();
        }
        return total_size;
    }

private:
    std::vector<KernelBlockType> blocks_;
};

static auto sparse_key_hasher = [](const uint64_t& sign) {
    // going to this shard sign always have same remainder of sign % shard_num,
    // we flip high and low bit to avoid hashmap bucket conflict probability.
    return std::hash<uint64_t>()(sign >> 32 | sign << 32);
};

template <typename OptType, typename ValueType>
class SparseKernelBlock {
public:
    // here must be gurantee that hashmap reserve bucket size is a prime number,
    // 15485863 is close to 16M, its mean that we can store sparse parameter size
    // in one single node is 16M * SPARSE_KERNEL_BLOCK_NUM, in which setting we can
    // supporting close to 5B parameter with 50 node run together without rehash.
    // TODO the initial bucket size maybe expose to user by a configure.
    SparseKernelBlock(const OptimizerBase* opt, int dimension)
        : values_(15485863, sparse_key_hasher)
        , dim_(dimension)
        , alloc_(ValueType::DynSizeof(dim_), 1 << 16) {
        values_.max_load_factor(0.75);
        opt_ = dynamic_cast<const OptType*>(opt);
        mutex_ = std::make_unique<std::mutex>();
    }

    SparseKernelBlock(SparseKernelBlock&) = delete;
    SparseKernelBlock& operator=(SparseKernelBlock&) = delete;

    SparseKernelBlock(SparseKernelBlock&& other)
        : opt_(other.opt_)
        , values_(std::move(other.values_))
        , mutex_(std::move(other.mutex_))
        , dim_(other.dim_)
        , alloc_(std::move(other.alloc_))
    { }

    SparseKernelBlock& operator=(SparseKernelBlock&& other) {
        opt_ = other.opt_;
        values_ = std::move(other.values_);
        mutex_ = std::move(other.mutex_);
        dim_ = other.dim_;
        alloc_ = std::move(other.alloc_);

        return this;
    }

    ~SparseKernelBlock() {
        for (const auto& iter : values_) {
            if (iter.second) {
                alloc_.deallocate(iter.second);
            }
        }
    }

    float* GetWeight(uint64_t sign) {
        const std::lock_guard<std::mutex> lock(*mutex_);

        auto inserted = values_.insert({sign, nullptr});
        if (inserted.second) {
            inserted.first->second = alloc_.allocate(dim_, opt_);
        }

        return inserted.first->second->Weight();
    }

    void Apply(uint64_t sign, SparseGradInfo& grad_info) {
        std::lock_guard<std::mutex> lock(*mutex_);
        auto iter = values_.find(sign);

        // must already meet and created in pull
        CHECK(iter != values_.end()) << " embedding of sign " << sign
            << " not create yet, something must be wrong";

        ValueType* value = iter->second;

        value->Apply(opt_, grad_info, dim_);
    }

    size_t Size() const {
        return values_.size();
    }

    friend std::ostream& operator<<(std::ostream& os, const SparseKernelBlock& block) {
        std::lock_guard<std::mutex> lock(*block.mutex_);

        os << "opt_name:" << block.opt_->Name() << std::endl;
        os << "dim:" << block.dim_ << std::endl;

        for (const auto& value : block.values_) {
            os << value.first << "\t";
            value.second->Serialize(os, block.dim_);
            os << std::endl;
        }

        return os;
    }

    friend std::istream& operator>>(std::istream& is, SparseKernelBlock& block) {
        std::lock_guard<std::mutex> lock(*block.mutex_);

        std::string opt_name;
        is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> opt_name;

        CHECK_EQ(opt_name, block.opt_->Name()) << "last trained model with optimizer is:" << opt_name
            << " but current model use:" << block.opt_->Name() << " instead."
            << " you must make sure that use same optimizer when incremental training";

        is.ignore(std::numeric_limits<std::streamsize>::max(), ':') >> block.dim_;

        uint64_t sign = 0;
        while (is >> sign) {
            ValueType* value = block.alloc_.allocate(block.dim_, block.opt_);
            value->DeSerialize(is, block.dim_);
            block.values_[sign] = value;
        }

        return is;
    }

    void ShowDecay() {
        for (auto& iter : values_) {
            ValueType* value = iter.second;
            value->ShowDecay(opt_);
        }
    }

private:
    const OptType* opt_ = nullptr;
    std::unordered_map<uint64_t, ValueType*, decltype(sparse_key_hasher)> values_;

    std::unique_ptr<std::mutex> mutex_;
    int dim_ = 0;

    Allocator<ValueType> alloc_;
};

template <typename KernelBlockType>
class SparseOptimizerKernel : public SparseOptimizerKernelBase {
public:
    SparseOptimizerKernel(const OptimizerBase* opt, int dimension) {
        assert(nullptr != opt);

        for (size_t i = 0; i < SPARSE_KERNEL_BLOCK_NUM; ++i) {
            blocks_.emplace_back(opt, dimension);
        }
    }

    ~SparseOptimizerKernel() = default;

    float* GetWeight(uint64_t sign) {
        int block_num = GetBlockId_(sign);
        return blocks_[block_num].GetWeight(sign);
    }

    void Apply(uint64_t sign, SparseGradInfo& grad_info) {
        int block_num = GetBlockId_(sign);
        blocks_[block_num].Apply(sign, grad_info);
    }

    void Serialized(const std::string& filepath) {
        std::vector<std::thread> threads;

        for (size_t i = 0; i < SPARSE_KERNEL_BLOCK_NUM; ++i) {
            threads.push_back(std::thread([this, i, &filepath]() {
                std::string file = filepath;
                file.append("/sparse_block_").append(std::to_string(i)).append(".gz");

                FileWriterSink writer_sink(file, FCT_ZLIB);

                boost::iostreams::stream<FileWriterSink> out_stream(writer_sink);

                out_stream << blocks_[i] << std::endl;
                out_stream.flush();
            }));
        }

        std::for_each(threads.begin(), threads.end(), [](std::thread& t) {
            t.join();
        });
    }

    void DeSerialized(const std::string& filepath) {
        std::vector<std::thread> threads;

        for (size_t i = 0; i < SPARSE_KERNEL_BLOCK_NUM; ++i) {
            threads.push_back(std::thread([this, i, &filepath]() {
                std::string file = filepath;
                file.append("/sparse_block_").append(std::to_string(i)).append(".gz");

                FileReaderSource reader_source(file, FCT_ZLIB);
                boost::iostreams::stream<FileReaderSource> in_stream(reader_source);

                in_stream >> blocks_[i];
            }));
        }

        std::for_each(threads.begin(), threads.end(), [](std::thread& t) {
            t.join();
        });
    }

    size_t KeyCount() const {
        size_t key_count = 0;
        for (size_t i = 0; i < SPARSE_KERNEL_BLOCK_NUM; ++i) {
            key_count += blocks_[i].Size();
        }

        return key_count;
    }

    void ShowDecay() {
        for (size_t i = 0; i < SPARSE_KERNEL_BLOCK_NUM; ++i) {
            blocks_[i].ShowDecay();
        }
    }

private:
    int GetBlockId_(uint64_t sign) {
        return sparse_key_hasher(sign) % SPARSE_KERNEL_BLOCK_NUM;
    }

private:
    std::vector<KernelBlockType> blocks_;
};

} // namespace tensornet {

#endif // !TENSORNET_OPTIMIZER_KERNEL_H_
