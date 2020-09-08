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

#ifndef TENSORNET_UTILITY_BLOCKING_QUEUE_H_
#define TENSORNET_UTILITY_BLOCKING_QUEUE_H_

#include <vector>
#include <queue>
#include <condition_variable>

namespace tensornet {

/*
 * BlockingQueue simply adapts a queue to provide thread-safe functionality.
 *
 * The internal queue is protected by a mutex for all operations to
 * ensure that no data races occur. In addition, threads attempting
 * to pop an item off of an empty queue using pop() will sleep until
 * an item has been pushed by another thread.
 *
 * To atomically push or pop multiple items in bulk, one can use lock()
 * and unlock() (or a std::unique_lock object) on the queue itself, and
 * other threads are guaranteed not to interfere.
 *
 * Queue must support push(), pop(), front(), empty(), and Queue(Queue&&).
 */
template <typename T, class Queue = std::queue<T>>
class BlockingQueue {
public:
    using value_type = T;

    BlockingQueue()
        : owning_thread_(std::thread::id())
    { }

    BlockingQueue(BlockingQueue&& other) noexcept
        : owning_thread_(std::thread::id()) {
        auto locker = other.get_lock();
        q = std::move(other.q);
    }

    ~BlockingQueue() = default;

    BlockingQueue(const BlockingQueue&) = delete;

    void push(T const& item) {
        auto locker = get_lock();

        // if the queue is empty, threads might be waiting to pop
        bool wasEmpty = empty_impl();
        q.push(item);
        if (wasEmpty) {
            // wake up a thread waiting to pop, if any
            cv.notify_one();
        }
    }

    void push(T&& item) {
        auto locker = get_lock();
        push_impl(std::forward<T>(item));

        // if the queue is empty, threads might be waiting to pop
        bool wasEmpty = empty_impl();
        q.push(std::forward<T>(item));
        if (wasEmpty) {
            // wake up a thread waiting to pop, if any
            cv.notify_one();
        }
    }

    // removes an item from the queue and returns it as an rvalue
    T pop() {
        auto locker = get_lock();
        return pop_impl();
    }
    T try_pop() {
        auto locker = get_lock();
        return try_pop_impl();
    }

    // removes all elements from the queue
    void clear() {
        auto locker = get_lock();
        clear_impl();
    }

    // returns true if the queue currently contains no elements
    bool empty() {
        auto locker = get_lock();
        return empty_impl();
    }

    // prevent and reallow modification of the queue
    void lock() {
        m.lock();
        owning_thread_.store(std::this_thread::get_id());
    }

    bool try_lock() {
        if (m.try_lock()) {
            owning_thread_.store(std::this_thread::get_id());
            return true;
        }
        return false;
    }

    void unlock() {
        owning_thread_.store(std::thread::id());
        m.unlock();
    }

    // returns true if the calling thread has locked the queue
    bool owns_lock() const {
        return owning_thread_.load() == std::this_thread::get_id();
    }

private:
    // If the calling thread already owns the lock, a lock initialized with
    // std::deferred_lock is returned.
    // Otherwise, the lock is initialized normally.
    std::unique_lock<BlockingQueue<T, Queue>> get_lock() {
        using lock_t = std::unique_lock<BlockingQueue<T, Queue>>;

        if (owns_lock()) {
            // the queue is already locked, do not relock it or unlock it on destruction
            return lock_t(*this, std::defer_lock);
        } else {
            // lock the queue
            return lock_t(*this);
        }
    }

    // All impl methods assume the queue is already locked.
    void push_impl(T const& item);
    void push_impl(T&& item);

    T pop_impl();
    T try_pop_impl();
    void clear_impl();
    bool empty_impl();

    // the underlying container
    Queue q;
    // protects the queue
    std::mutex m;
    // notifies threads waiting on pop() that the queue is no longer empty
    std::condition_variable_any cv;
    // stores the thread id of the current owner
    std::atomic<std::thread::id> owning_thread_;
};

/*
 * A lightweight adaptor that makes std::priority_queue fit the std::queue interface
 * required by BlockingQueue. Because only named objects of this type will be used,
 * polymorphism (especially in the destructor) is not necessary.
 */
template <class PriorityQueue>
class PriorityQueueWrapper : public PriorityQueue
{
public:
    typedef typename PriorityQueue::value_type value_type;
    const value_type& front() const;
};

/*
 * Convenience typedef for using a priority queue in a BlockingQueue.
 */
template <typename T, class PriorityQueue = std::priority_queue<T>>
using BlockingPriorityQueue = BlockingQueue<T, PriorityQueueWrapper<PriorityQueue>>;


template <typename T, class Queue>
T BlockingQueue<T, Queue>::pop_impl()
{
    // assuming *this is already locked, create a std::unique_lock for cv.wait()
    auto locker = std::unique_lock<typename std::remove_reference<decltype(*this)>::type>(*this,
            std::adopt_lock);

    // wait until the queue is not empty
    cv.wait(locker, [this] () {
        return !empty_impl();
    });

    // disassociate the queue from the lock without unlocking it
    locker.release();

    return try_pop_impl();
}

template <typename T, class Queue>
T BlockingQueue<T, Queue>::try_pop_impl()
{
    if (empty_impl()) {
        throw std::out_of_range("BlockingQueue is empty.");
    }
    // pop the front item off the queue and return it
    auto temp = std::move(q.front());
    q.pop();
    return std::move(temp);
}

template <typename T, class Queue>
void BlockingQueue<T, Queue>::clear_impl()
{
    // assuming *this is already locked, pop all of the queue's elements
    while (!empty_impl()) {
        pop_impl();
    }
}

template <typename T, class Queue>
bool BlockingQueue<T, Queue>::empty_impl()
{
    return q.empty();
}


// PriorityQueue Wrapper Implementation - simply adds front() method that uses top()
template <class PriorityQueue>
typename PriorityQueueWrapper<PriorityQueue>::value_type const&
PriorityQueueWrapper<PriorityQueue>::front() const
{
    return PriorityQueue::top();
}

}  // namespace tensornet

#endif  // TENSORNET_UTILITY_BLOCKING_QUEUE_H_
