#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
    bool finished = false;

public:
    void push(T val) {
        {
            std::lock_guard<std::mutex> lk(m);
            q.push(std::move(val));
        }
        cv.notify_one();
    }

    bool pop(T& val) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this]{ return !q.empty() || finished; });
        if (q.empty() && finished) {
            return false;
        }
        val = std::move(q.front());
        q.pop();
        return true;
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(m);
            finished = true;
        }
        cv.notify_all();
    }
};
