#pragma once

#include "frame.h"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <unordered_map>

/** Збирає кадри з паралельних парсерів і видає їх GPU у порядку зростання індекса. */
class OrderedFrameBuffer {
public:
    void push(size_t index, Frame frame) {
        std::lock_guard<std::mutex> lk(mu_);
        ready_.insert_or_assign(index, std::move(frame));
        cv_.notify_all();
    }

    void set_fail() {
        std::lock_guard<std::mutex> lk(mu_);
        failed_ = true;
        cv_.notify_all();
    }

    bool failed() const {
        std::lock_guard<std::mutex> lk(mu_);
        return failed_;
    }

    /** Блокує до появи кадру index або глобальної помилки. Повертає false при fail. */
    bool wait_take(size_t index, Frame& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return ready_.count(index) != 0 || failed_; });
        if (failed_) return false;
        out = std::move(ready_[index]);
        ready_.erase(index);
        return true;
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<size_t, Frame> ready_;
    bool failed_ = false;
};
