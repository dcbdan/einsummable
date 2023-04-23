#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/execute.h"

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
  int num_processors = 1; // TODO: set with mpi world size
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

  graph_t graph = three_dimensional_matrix_multiplication(
    pi,pj,pk, di,dj,dk, num_processors);

  graph.print();

  vector<char> _line(40, '/');
  std::string line(_line.begin(), _line.end());
  std::cout << line << std::endl << std::endl;

  auto [input_blocks, output_blocks, taskgraph] = taskgraph_t::make(graph);

  taskgraph.print();

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

  std::cout << line << std::endl << std::endl;

  execute(taskgraph, tensors);

  for(auto const& [gid, out_blocks]: output_blocks) {
    for(auto const& out: out_blocks.get()) {
      std::cout << out << " " << tensors.at(out) << std::endl;
    }
  }

  // TODO: verify results are correct to reference over the graph
}


