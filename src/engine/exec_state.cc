#include "exec_state.h"
#include "gpu/exec_nodes.h"
#include <unordered_set> 

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
    auto init_time = std::chrono::high_resolution_clock::now();

    num_remaining = num_nodes;

    vector<int> ready_to_run_;
    num_deps_remaining.reserve(num_nodes);
    wait_start_time.reserve(num_nodes);
    total_mem_wait_time = std::chrono::milliseconds(0);
    all_inns.reserve(num_nodes);
    auto& last_manager = std::dynamic_pointer_cast<resource_manager_t>(resource_manager)->managers.back();
    streampool_manager = std::dynamic_pointer_cast<streampool_manager_t>(last_manager);
    desc_vector.reserve(num_nodes);
    for (int id = 0; id != num_nodes; ++id) {
        int num_deps = g.nodes[id].inns.size();
        wait_start_time.push_back(std::chrono::milliseconds(0));
        num_deps_remaining.push_back(num_deps);
        desc_vector.push_back(0);
        std::unordered_set<int> inns_set(g.nodes[id].inns.begin(), g.nodes[id].inns.end());
        all_inns.push_back(inns_set);
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
    init_time = std::chrono::high_resolution_clock::now();
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

void exec_state_t::start_timer_if_condition(int device, int out_id, int kind){
    if (streampool_manager->all_streams_vacant(device)) {
            //if non of the streams are being used, then we start the counter
            auto waitstart = std::chrono::high_resolution_clock::now();
            wait_start_time[out_id] = std::chrono::duration_cast<std::chrono::milliseconds>(waitstart - init_time);
            desc_vector[out_id] = kind;
        }
}


void exec_state_t::decrement_outs(int id)
{
    auto const& node = exec_graph.nodes[id];
    for (auto const& out_id : node.outs) {
        int& cnt = num_deps_remaining[out_id];
        cnt--;
        all_inns[out_id].erase(id);
        if (cnt == 1) { 
            if (all_inns[out_id].size() != 1) {
                throw std::runtime_error("all_inns size does not match num_deps_remaining");
            }
            //if the only inns left is load then we start the counter for the out_id
            auto const& id_left = *(all_inns[out_id].begin());
            auto const& node_left = exec_graph.nodes[id_left];
            if (exec_graph_t::op_base_t::is_gpu_load(node_left.op)) {
                int dev = std::dynamic_pointer_cast<gpu_load_t>(node_left.op)->device;
                start_timer_if_condition(dev, out_id, 1);
            }
        }
        if (cnt == 0) {
            if (wait_start_time[out_id] != std::chrono::milliseconds(0)) {
                auto waitend = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(waitend - init_time - wait_start_time[out_id]);
                std::cout << "duration for node " << id << " to node " << out_id << " of type " << desc_vector[out_id] << " is " << duration.count() << std::endl;
                total_mem_wait_time += duration;
            }
            auto const& out_node = exec_graph.nodes[out_id];
            auto out_op = out_node.op;
            ready_to_run.push(out_id);
            // if evict node got ready, then the alloc(apply)/load that depends on the evict would come 
            if (exec_graph_t::op_base_t::is_gpu_evict(out_op)) {
                for (auto const& evict_out_id: out_node.outs) {
                    auto const& evict_out_node = exec_graph.nodes[evict_out_id];
                    if (!exec_graph_t::op_base_t::is_gpu_load(evict_out_node.op)) {
                        // evict could only be pointing into alloc or load. We want to find the alloc it's depending on
                        // if it's alloc, then we need to go to the next layer (the actual apply node)
                        auto const& alloc_node = exec_graph.nodes[evict_out_id];
                        if (alloc_node.outs.size() != 1) {
                            throw std::runtime_error("out of alloc node should only be sized 1");
                        }
                        bool all_ready = true;
                        for (auto const& alloc_inn_id: alloc_node.inns) {
                            //loop through the alloc node to see if all other nodes are ready
                            // if all other nodes are ready, then start time counter for apply_id below (run the code below only if all other nodes ready)
                            if (num_deps_remaining[alloc_inn_id] != 0) {
                                all_ready = false;
                            }
                        }
                        if (all_ready == true) {
                            const int& apply_id = alloc_node.outs.at(0);
                            int apply_node_cnt = num_deps_remaining[apply_id];
                            // Here I don't have to decreement the count for apply_node, because it will get its number decremented when we actually get to that dummy alloc node
                            if (apply_node_cnt == 1) {
                                // if this alloc is the only thing blocking way, start time counter for apply_id
                                int dev = exec_graph_t::op_base_t::get_device(exec_graph.nodes[apply_id].op);
                                start_timer_if_condition(dev, apply_id, 2);
                            }
                        }
                    } else {
                        // if it's a load after evict, then start the timer for the load node directly is fine
                        int dev = exec_graph_t::op_base_t::get_device(evict_out_node.op);
                        start_timer_if_condition(dev, evict_out_id, 3);
                    }
                }
                
            } else if (exec_graph_t::op_base_t::is_gpu_load(out_op)) {
                //unary op, start counter
                for (auto const& load_out_id: out_node.outs) {
                    auto const& load_out_node = exec_graph.nodes[load_out_id];
                    if (load_out_node.inns.size() == 1) {
                        // Make sure if the out node of load is unary. 
                        // If is unary, then start the counter once load is ready (already ready here)
                        int dev = exec_graph_t::op_base_t::get_device(load_out_node.op);
                        start_timer_if_condition(dev, load_out_id, 4);
                    } 
                }
            }
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
                std::lock_guard<std::mutex> lock(mutex_io_time);
                io_start_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(launchstart-init_time));
                io_end_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(launchend-init_time));
                io_time_total += launchduration;
            } else {
                std::lock_guard<std::mutex> lock(mutex_kernel_time);
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
