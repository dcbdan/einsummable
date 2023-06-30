#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/execute.h"
#include "../src/execution/cpu/mpi_class.h"

#define ROUT(x) if(mpi.this_rank == 0) { std::cout << "(in cpu exec) " << x << std::endl; }

void usage() {
  std::cout << "Usage: pi pj pk di dj dk\n"
            << "\n"
            << "Multiply a ('di'*'pi', 'dj'*'pj') matrix with\n"
            << "         a ('dj'*'pj', 'dk'*'pk') marrix\n"
            << "using the 3D matrix multiply algorithm.\n"
            << "\n"
            << "The multiply occurs over a virtual grid of\n"
            << "'pi'*'pj'*'pk' processors mapped to\n"
            << "mpi world size physical processors\n";
}

int main(int argc, char** argv) {
  if(argc != 7) {
    usage();
    return 1;
  }

  int pi, pj, pk;
  uint64_t di, dj, dk;
  try {
    pi             = parse_with_ss<int>(     argv[1]);
    pj             = parse_with_ss<int>(     argv[2]);
    pk             = parse_with_ss<int>(     argv[3]);
    di             = parse_with_ss<uint64_t>(argv[4]);
    dj             = parse_with_ss<uint64_t>(argv[5]);
    dk             = parse_with_ss<uint64_t>(argv[6]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }

  // auto settings = settings_t::default_settings();
  int num_threads_per_node = 4;
  settings_t execute_settings {
    
    // .num_apply_runner = num_threads_per_node,
    // .num_touch_runner = 2,
  
    .num_kernel_runner = num_threads_per_node,
    .num_send_runner  = 1,
    .num_recv_runner  = 1,
    .num_apply_kernel_threads = 1
  };

  mpi_t mpi(argc, argv);

  int num_processors = mpi.world_size;

  dtype_t dtype = default_dtype();

  auto g = three_dimensional_matrix_multiplication(pi,pj,pk, di,dj,dk, num_processors);
  graph_t const& graph = g.graph;
  auto pls = g.get_placements();

  //if(mpi.this_rank == 0) {
  //  graph.print();
  //}

  vector<char> _line(40, '/');
  std::string line(_line.begin(), _line.end());

  //if(mpi.this_rank == 0) {
  //  std::cout << line << std::endl << std::endl;
  //}

  auto [input_blocks, output_blocks, taskgraph] = taskgraph_t::make(graph, pls);

  //if(mpi.this_rank == 0) {
  //  taskgraph.print();
  //}

  // Initialize the input tensors to all ones
  map<int, buffer_t> tensors;
  for(auto const& [gid, inn_blocks]: input_blocks) {
    for(auto const& inn: inn_blocks.get()) {
      auto const& node = taskgraph.nodes[inn];
      auto const& [rank, size] = node.op.get_input();
      if(mpi.this_rank == rank) {
        buffer_t buffer = make_buffer(size);
        dbuffer_t(dtype, buffer).random("-0.0003", "0.003");
        tensors.insert({inn, buffer});
      }
    }
  }

  //if(mpi.this_rank == 0) {
  //  std::cout << line << std::endl << std::endl;
  //}

  {
    mpi.barrier();
    raii_print_time_elapsed_t gremlin("3D Matmul Time");
    execute(taskgraph, execute_settings, mpi, tensors);
    mpi.barrier();
  }
}


