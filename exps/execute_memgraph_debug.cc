#include "../src/engine/exec_graph.h"
#include "../src/base/buffer.h"
#include "../src/einsummable/memgraph.h"
#include "../src/engine/communicator.h"
#include "../src/engine/cpu/workspace_manager.h"
#include "../src/engine/managers.h"
#include "../src/engine/exec_state.h"
#include "../src/engine/notifier.h"
#include "../src/engine/cpu/exec_nodes.h"

int main(int argc, char** argv) {

    int expected_argc = 5;
    if(argc < expected_argc) {
        std::cout << "Need more arg.\n" << std::endl;
        return 1;
    }

    string addr_zero = parse_with_ss<string>(argv[1]);
    bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
    int world_size = parse_with_ss<int>(argv[3]);

    uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
    uint64_t GB = 1000000000;
    mem_size *= GB;
    buffer_t mem = make_buffer(mem_size);

    // set up the communicator
    int num_channels_per_move = 1;
    int num_channels = 8;
    communicator_t comm(addr_zero, is_rank_zero, world_size, num_channels);

    // read the memgraph here
    memgraph_t memgraph;
    {
      std::stringstream buffer;
      std::ifstream file("mg.proto");
      buffer << file.rdbuf();
      memgraph = memgraph_t::from_wire(buffer.str());
    }

    comm.barrier();

    // create threadpool here
    threadpool_t threadpool(12);
    int n_threads = threadpool.num_runners();
    int this_rank = comm.get_this_rank();

    cpu_kernel_executor_t kernel_executor;

    exec_graph_t graph = exec_graph_t::make_cpu_exec_graph(
        memgraph,
        this_rank,
        kernel_executor,
        num_channels_per_move);

    vector<rm_ptr_t> managers;
    managers.emplace_back(new cpu_workspace_manager_t());
    managers.emplace_back(new group_manager_t());
    managers.emplace_back(new global_buffers_t(mem->raw()));

    rm_ptr_t rcm_ptr(new recv_channel_manager_t(comm));
    recv_channel_manager_t& rcm = *static_cast<recv_channel_manager_t*>(rcm_ptr.get());
    managers.emplace_back(new notifier_t(comm, rcm));
    managers.emplace_back(new send_channel_manager_t(comm, n_threads-1));
    managers.push_back(rcm_ptr);

    managers.emplace_back(new threadpool_manager_t(threadpool));

    rm_ptr_t resource_manager(new resource_manager_t(std::move(managers)));

    exec_state_t state(graph, resource_manager, exec_state_t::priority_t::given, this_rank);

    state.event_loop();
}