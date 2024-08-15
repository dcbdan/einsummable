#include "exec_state.h"
#include "gpu/exec_nodes.h"

// #define EXEC_STATE_PRINT
#define EXEC_STATE_COUNTDOWN

#include <fstream>
int _filecnt = 0;

exec_state_t::exec_state_t(exec_graph_t const&      g,
                           rm_ptr_t                 r,
                           exec_state_t::priority_t p,
                           int                      this_rank)
    : exec_graph(g), resource_manager(r), ready_to_run(this), this_rank(this_rank), io_time_total(0), kernel_time_total(0)
{
#ifdef EXEC_STATE_PRINT
    DLINEOUT("this rank is " << this_rank << " | filecnt " << _filecnt);
    out = std::ofstream("exec_state_out_cnt" + write_with_ss(_filecnt++) + "_rank" +
                        write_with_ss(this_rank));

    for (int i = 0; i != g.nodes.size(); ++i) {
        out << i << ": ";
        g.nodes[i].print(out);
        out << std::endl;
    }
#endif

    int num_nodes = exec_graph.nodes.size();

    num_remaining = num_nodes;

    vector<int> ready_to_run_;
    num_deps_remaining.reserve(num_nodes);
    for (int id = 0; id != num_nodes; ++id) {
        int num_deps = g.nodes[id].inns.size();

        num_deps_remaining.push_back(num_deps);

        if (num_deps == 0) {
            ready_to_run_.push_back(id);
        }
    }

    priorities.reserve(num_nodes);
    if (p == priority_t::given) {
        // DOUT("using given priorities");
        for (auto const& node : exec_graph.nodes) {
            priorities.push_back(node.priority);
        }
    } else if (p == priority_t::bfs) {
        // DOUT("using bfs priorities");
        vector<int> deps = num_deps_remaining;
        vector<int> order = ready_to_run_;
        order.reserve(num_nodes); // this prevents iterators from being invalidated
        for (auto iter = order.begin(); iter != order.end(); ++iter) {
            int         id = *iter;
            auto const& node = exec_graph.nodes[id];
            for (auto const& out_id : node.outs) {
                int& d = deps[out_id];
                d--;
                if (d == 0) {
                    order.push_back(out_id);
                }
            }
        }
        if (order.size() != num_nodes) {
            throw std::runtime_error("should not happen: bfs");
        }
        priorities.resize(num_nodes);
        for (int idx = 0; idx != num_nodes; ++idx) {
            int const& id = order[idx];
            priorities[id] = idx;
        }
    } else if (p == priority_t::dfs) {
        // DOUT("using dfs priorities");
        int cnt = 0;
        priorities.resize(num_nodes);
        vector<int> deps = num_deps_remaining;
        vector<int> processing = ready_to_run_;
        while (processing.size() > 0) {
            int id = processing.back();
            processing.pop_back();

            priorities[id] = cnt++;

            auto const& node = exec_graph.nodes[id];
            for (auto const& out_id : node.outs) {
                int& d = deps[out_id];
                d--;
                if (d == 0) {
                    processing.push_back(out_id);
                }
            }
        }
    } else if (p == priority_t::random) {
        DOUT("using random priorities");
        for (int id = 0; id != num_nodes; ++id) {
            priorities.push_back(runif(1000));
        }
    } else {
        throw std::runtime_error("this priority type is not implemented");
    }

    // only now that the priorites is set can we safely use
    // ready_to_run as a queue proper
    for (auto const& id : ready_to_run_) {
        ready_to_run.push(id);
    }
}

void exec_state_t::event_loop()
{
#ifdef EXEC_STATE_COUNTDOWN
    int decrement_print_at = 20000;
    if (num_remaining > decrement_print_at) {
        DOUT("eventloop num remaining: " << num_remaining);
    }
    int print_at = num_remaining - decrement_print_at;
#endif
    vector<int> processing;
    while (true) {
        while (processing.size() > 0) {
            int id = processing.back();
            processing.pop_back();

#ifdef EXEC_STATE_PRINT
            out << "finished " << id << std::endl;
#endif

            auto iter = is_running.find(id);

            resource_manager->release(iter->second);

            is_running.erase(iter);

            decrement_outs(id);

            num_remaining--;
        }

#ifdef EXEC_STATE_COUNTDOWN
        if (num_remaining < print_at) {
            DOUT("eventloop num remaining: " << num_remaining);
            print_at = num_remaining - decrement_print_at;
        }
#endif
        if (num_remaining == 0) {
            return;
        }

        {
            queue_t failed(this);
            while (ready_to_run.size() > 0) {
                int id = ready_to_run.top();
                ready_to_run.pop();
                if (try_to_launch(id)) {
                    // started id
#ifdef EXEC_STATE_PRINT
                    out << "started " << id << std::endl;
#endif
                } else {
                    failed.push(id);
                }
            }
            ready_to_run = failed;
        }

        // for each thing in ready to run:
        //   try to grab resource
        //   launch
        //     > inside call back, release resource
        //       and add to just_completed
        std::unique_lock lk(m_notify);
        cv_notify.wait(lk, [&, this] {
            if (just_completed.size() > 0) {
                processing = just_completed;
                just_completed.resize(0);
                return true;
            } else {
                return false;
            }
        });
    }
}

void exec_state_t::decrement_outs(int id)
{
    auto const& node = exec_graph.nodes[id];
    for (auto const& out_id : node.outs) {
        int& cnt = num_deps_remaining[out_id];
        cnt--;
        if (cnt == 0) {
            ready_to_run.push(out_id);
        }
    }
}

bool exec_state_t::try_to_launch(int id)
{
    auto const&    node = exec_graph.nodes[id];
    desc_ptr_t     resource_desc = node.resource_description();
    resource_ptr_t resources = resource_manager->try_to_acquire(resource_desc);
    if (resources) {
        auto launchstart = std::chrono::high_resolution_clock::now();
        auto callback = [this, id, launchstart] {
            auto launchend = std::chrono::high_resolution_clock::now();
            auto launchduration = std::chrono::duration_cast<std::chrono::milliseconds>(launchend - launchstart);
            auto const&    node = exec_graph.nodes[id];
            if (dynamic_cast<const gpu_load_t*>(node.op.get()) || dynamic_cast<const gpu_evict_t*>(node.op.get())) {
                io_time_total += launchduration;
            } else {
                kernel_time_total += launchduration;
            }
            {
                std::unique_lock lk(m_notify);
                this->just_completed.push_back(id);
            }

            cv_notify.notify_one();
        };
        // DOUT("launching " << id);
        node.launch(resources, callback);

        is_running.insert({id, resources});

        return true;
    } else {
        return false;
    }
}
