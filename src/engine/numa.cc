#include "numa.h"

#include <sched.h>
#include <thread>

numa_info_t::numa_info_t() : num_numa(0)
{
    cpu_set_t orig_cpu_set;
    if (sched_getaffinity(0, sizeof(orig_cpu_set), &orig_cpu_set) != 0) {
        throw std::runtime_error("could not get orig cpu set");
    }

    unsigned int which_cpu;
    unsigned int which_numa;
    unsigned int this_numa;

    // set numa
    if (getcpu(&which_cpu, &this_numa) != 0) {
        throw std::runtime_error("getcpu fail");
    }

    int nc = std::thread::hardware_concurrency();

    for (int i = 0; i != nc; ++i) {
        pin_to_thread(i);

        unsigned int which_cpu;
        if (getcpu(&which_cpu, &which_numa) != 0) {
            throw std::runtime_error("getcpu fail");
        }
        num_numa = std::max(num_numa, which_numa);

        if (which_numa == this_numa) {
            which.push_back(which_cpu);
        }
    }

    if (sched_setaffinity(0, sizeof(orig_cpu_set), &orig_cpu_set) != 0) {
        throw std::runtime_error("could not reset thread");
    }

    num_numa += 1;
}

void numa_info_t::pin_to_thread(int i)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(i, &cpu_set);
    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        throw std::runtime_error("could not pin thread");
    }
}

void numa_info_t::unpin()
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    int nc = std::thread::hardware_concurrency();
    for (int i = 0; i != nc; ++i) {
        CPU_SET(i, &cpu_set);
    }
    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        throw std::runtime_error("could not (un)pin thread");
    }
}

void numa_info_t::print()
{
    unsigned int which_cpu;
    unsigned int which_numa;
    int          nc = std::thread::hardware_concurrency();
    for (int i = 0; i != nc; ++i) {
        pin_to_thread(i);
        if (getcpu(&which_cpu, &which_numa) != 0) {
            throw std::runtime_error("getcpu fail");
        }
        if (i != which_cpu) {
            throw std::runtime_error("i != which_cpu: print");
        }
        DOUT("cpu " << which_cpu << " lives on numa " << which_numa);
    }
}

void numa_info_t::pin_to_this_numa_thread(int n) const
{
    if (n > which.size()) {
        throw std::runtime_error("cannot exceed hardward concurrency on this numa");
    }
    auto const& w = which[n];
    pin_to_thread(w);
}

void numa_info_t::pin_to_this_numa() const
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (auto const& i : which) {
        CPU_SET(i, &cpu_set);
    }
    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        throw std::runtime_error("could not pin thread to this numa");
    }
}

numa_info_t const& get_numa_info()
{
    static numa_info_t info;
    return info;
}
