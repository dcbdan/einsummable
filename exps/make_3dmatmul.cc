#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"

#include <fstream>

void usage() {
  std::cout << "Usage: pi pj pk di dj dk num_processors\n"
            << "\n"
            << "Multiply a ('di'*'pi', 'dj'*'pj') matrix with\n"
            << "         a ('dj'*'pj', 'dk'*'pk') marrix\n"
            << "using the 3d matrix multiply algorithm.\n"
            << "\n"
            << "The multiply occurs over a virtual grid of\n"
            << "'pi'*'pj'*'pk' processors mapped to\n"
            << "'num_processors' physical processors\n";
}

int main(int argc, char** argv) {
  if(argc != 8) {
    usage();
    return 1;
  }

  int pi, pj, pk;
  uint64_t di, dj, dk;
  int num_processors;
  try {
    pi             = parse_with_ss<int>(     argv[1]);
    pj             = parse_with_ss<int>(     argv[2]);
    pk             = parse_with_ss<int>(     argv[3]);
    di             = parse_with_ss<uint64_t>(argv[4]);
    dj             = parse_with_ss<uint64_t>(argv[5]);
    dk             = parse_with_ss<uint64_t>(argv[6]);
    num_processors = parse_with_ss<int>(     argv[7]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }

  graph_constructor_t g = three_dimensional_matrix_multiplication(
    pi,pj,pk, di,dj,dk, num_processors);

  {
    std::ofstream f("g.gv");
    g.graph.print_graphviz(f);
    DOUT("printed g.gv");
  }

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  vector<char> line(40, '/');
  std::cout << std::string(line.begin(), line.end()) << std::endl;

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }
}


