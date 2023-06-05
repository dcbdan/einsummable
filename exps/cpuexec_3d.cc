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
            << "using the 3d matrix multiply algorithm.\n"
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

  auto settings = settings_t::default_settings();

  mpi_t mpi(argc, argv);

  int num_processors = mpi.world_size;

  auto g = three_dimensional_matrix_multiplication(
    pi,pj,pk, di,dj,dk, num_processors);
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
      uint64_t const& size = node.op.get_input().size;
      buffer_t buffer = std::make_shared<buffer_holder_t>(size);
      buffer->ones();
      tensors.insert({inn, buffer});
    }
  }

  //if(mpi.this_rank == 0) {
  //  std::cout << line << std::endl << std::endl;
  //}

  {
    mpi.barrier();
    raii_print_time_elapsed_t gremlin("3D Matmul Time");
    execute(taskgraph, settings, mpi, tensors);
    mpi.barrier();
  }

  //for(int rank = 0; rank != mpi.world_size; ++rank) {
  //  mpi.barrier();
  //  if(rank == mpi.this_rank) {
  //    for(auto const& [gid, out_blocks]: output_blocks) {
  //      for(auto const& out: out_blocks.get()) {
  //        auto out_loc = taskgraph.nodes[out].op.output_loc();
  //        if(out_loc == mpi.this_rank) {
  //          std::cout << "rank " << rank << " | " << out << " " << tensors.at(out) << std::endl;
  //        }
  //      }
  //    }
  //  }
  //}

  // TODO: verify results are correct to reference over the graph
}

