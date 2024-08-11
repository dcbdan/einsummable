#pragma once
#include "../../base/setup.h"

#include "../base.h"

#include "../../engine/threadpool.h"
#include "../../engine/exec_state.h"

#include "../../engine/cpu/kernel_executor.h"
#include "../../engine/cpu/tg/data_manager.h"

struct cpu_tg_server_t : server_dist_base_t {
    cpu_tg_server_t(communicator_t& c_,
                    uint64_t        max_memory_usage,
                    int             num_threads,
                    int             num_threads_per_contraction = 1,
                    int             num_channels_per_move = 1)
        : server_dist_base_t(c_),
          max_memory_usage(max_memory_usage),
          threadpool(num_threads),
          num_threads_per_contraction(num_threads_per_contraction),
          num_channels_per_move(num_channels_per_move)
    {
        if (num_channels_per_move > num_threads) {
            throw std::runtime_error("invalid num channels per move");
        }

        // TODO: verify that num_channels_per_move is the same on all nodes
    }

    int get_num_threads() const
    {
        return threadpool.num_runners();
    }

    void local_insert_tensors(map<int, tuple<int, buffer_t>> data);

    void local_erase_tensors(vector<int> const& tids);

    bool make_parallel_partialize_groups() const
    {
        return false;
    }

    threadpool_t* get_cpu_threadpool()
    {
        return &threadpool;
    }

protected:
    void execute_tg_server(taskgraph_t const& taskgraph, map<string, scalar_t> const& scalar_vars);
    void execute_tg_client();

    void remap_server(remap_relations_t const& remap_relations);
    void remap_client();

    int local_get_max_tid() const;

    // return a location that exists at this compute-node
    int local_candidate_location() const;

    int loc_to_compute_node(int loc) const;

    buffer_t local_copy_data(int tid);

private:
    void _execute_tg(taskgraph_t const& taskgraph, map<string, scalar_t> const& scalar_vars = {});

    void _remap(remap_relations_t const& remap_relations);

private:
    uint64_t max_memory_usage;

    map<int, buffer_t> local_data;

    cpu_kernel_executor_t kernel_executor;

    threadpool_t threadpool;

    int num_threads_per_contraction;

    int num_channels_per_move;
};
