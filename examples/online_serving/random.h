#ifndef TENSORNET_UTILITY_RANDOM_H_
#define TENSORNET_UTILITY_RANDOM_H_

#include <atomic>
#include <random>
#include <ctime>

namespace tensornet {

inline double current_realtime() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

inline std::default_random_engine& local_random_engine() {
    struct engine_wrapper_t {
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::atomic<unsigned long> x(0);
            std::seed_seq sseq = {x++, x++, x++, (unsigned long)(current_realtime() * 1000)};
            engine.seed(sseq);
        }
    };

    thread_local engine_wrapper_t r;
    return r.engine;
}

template<class T = double>
std::uniform_real_distribution<T>& local_uniform_real_distribution() {
    thread_local std::uniform_real_distribution<T> distr;
    DCHECK(distr.a() == 0.0 && distr.b() == 1.0);
    return distr;
}

template<class T = double>
T uniform_real() {
    return local_uniform_real_distribution<T>()(local_random_engine());
}

template<class T = double>
T uniform_real(T a, T b) {
    if (a == b) {
        return a;
    }

    return (T)(a + uniform_real<T>() * (b - a));
}

} // namespace tensornet

#endif // TENSORNET_UTILITY_RANDOM_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
