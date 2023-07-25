#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <ranges>
#include <thread>

#include "KAS/Utils/Common.hpp"


namespace kas {

namespace detail {

template<typename Derived, typename T, typename V>
class ThreadPoolBase {
    using Result = std::conditional_t<std::is_void_v<V>, char, V>;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::condition_variable cvReady;
    std::queue<T> queue;
    std::vector<Result> results;
    std::vector<std::jthread> workers;
    std::size_t submitted = 0;
    std::size_t completed = 0;

    template<std::convertible_to<T> U>
    void post(std::unique_lock<std::mutex>& guard, U&& task) {
        queue.push(std::forward<U>(task));
        ++submitted;
    }

public:
    template<typename F>
    requires std::invocable<F, Derived&, T>
    ThreadPoolBase(std::size_t numWorkers, F&& f) {
        workers.reserve(numWorkers);
        for (std::size_t i = 0; i < numWorkers; ++i) {
            workers.emplace_back([this, f = std::forward<F>(f)](std::stop_token stopToken) {
                while (!stopToken.stop_requested()) {
                    std::optional<T> task;
                    {
                        std::unique_lock lock { mutex };
                        if(cv.wait(lock, stopToken, [this] { return !queue.empty(); })) {
                            task.emplace(std::move(queue.front()));
                            queue.pop();
                        }
                    }
                    if (task.has_value()) {
                        f(static_cast<Derived&>(*this), std::move(task.value()));
                        std::unique_lock lock { mutex };
                        ++completed;
                        if (completed == submitted) {
                            lock.unlock();
                            cvReady.notify_all();
                        }
                    }
                }
            });
        }
    }
    ThreadPoolBase(const ThreadPoolBase&) = delete;
    ThreadPoolBase(ThreadPoolBase&&) = delete;

    // Async.
    template<std::convertible_to<T> U>
    void add(U&& task) {
        std::unique_lock lock { mutex };
        post(lock, std::forward<U>(task));
        lock.unlock();
        cv.notify_one();
    }
    // Async.
    template<std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, T>
    void addMultiple(R&& tasks) {
        std::unique_lock lock { mutex };
        for (auto&& task: tasks) {
            post(lock, std::forward<decltype(task)>(task));
        }
        lock.unlock();
        cv.notify_all();
    }

    // Sync. Returns the completed number of tasks during this.
    template<std::convertible_to<T> U>
    std::size_t addSync(U&& task) {
        std::unique_lock lock { mutex };
        std::size_t currentCompleted = completed;
        post(lock, std::forward<U>(task));
        cv.notify_one();
        cvReady.wait(lock, [this] { return completed == submitted; });
        return completed - currentCompleted;
    }
    // Sync. Returns the completed number of tasks during this.
    template<std::ranges::input_range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, T>
    std::size_t addMultipleSync(R&& tasks) {
        std::unique_lock lock { mutex };
        std::size_t currentCompleted = completed;
        for (auto&& task: tasks) {
            post(lock, std::forward<decltype(task)>(task));
        }
        cv.notify_all();
        cvReady.wait(lock, [this] { return completed == submitted; });
        return completed - currentCompleted;
    }

    template<typename S>
    void pushResult(S&& result) {
        std::scoped_lock lock { mutex };
        results.push_back(std::forward<S>(result));
    }
    template<typename... Args>
    void emplaceResult(Args&&... args) {
        std::scoped_lock lock { mutex };
        results.emplace_back(std::forward<Args>(args)...);
    }
    std::vector<Result> dumpResults() {
        std::scoped_lock lock { mutex };
        return std::move(results);
    }

    ~ThreadPoolBase() {
        for (auto& worker: workers) {
            worker.request_stop();
        }
        workers.clear();
        if (submitted != completed) {
            KAS_WARNING("The thread pool is destroyed before all tasks are completed. (submitted == {}, completed == {})", submitted, completed);
        }
    }
};

} // namespace detail

template<typename T, typename V = void>
class ThreadPool: private detail::ThreadPoolBase<ThreadPool<T, V>, T, V> {
    friend class detail::ThreadPoolBase<ThreadPool, T, V>;
    using Super = detail::ThreadPoolBase<ThreadPool, T, V>;
public:
    using Super::Super;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    using Super::add;
    using Super::addMultiple;
    using Super::addSync;
    using Super::addMultipleSync;
    using Super::pushResult;
    using Super::emplaceResult;
    using Super::dumpResults;
};

template<typename V>
class ThreadPool<void, V>: private detail::ThreadPoolBase<ThreadPool<void, V>, char, V> {
    friend class detail::ThreadPoolBase<ThreadPool<void, V>, char, V>;
    using Super = detail::ThreadPoolBase<ThreadPool<void, V>, char, V>;
    static decltype(auto) Repeat(std::size_t count) {
        return std::views::iota(static_cast<std::size_t>(0), count) | std::views::transform([](std::size_t) { return ' '; });
    }

public:
    template<typename F>
    requires std::invocable<F, ThreadPool&>
    ThreadPool(std::size_t numWorkers, F&& f):
        Super(numWorkers, [f = std::forward<F>(f)](ThreadPool& pool, char) { f(pool); }) {}
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    void add() {
        Super::add(' ');
    }
    void addMultiple(std::size_t count) {
        Super::addMultiple(Repeat(count));
    }
    std::size_t addSync() {
        return Super::addSync(' ');
    }
    std::size_t addMultipleSync(std::size_t count) {
        return Super::addMultipleSync(Repeat(count));
    }
    using Super::pushResult;
    using Super::emplaceResult;
    using Super::dumpResults;
};

} // namespace kas
