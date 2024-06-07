#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/memgraph.h"

#include <fstream>

#include <fstream>

void usage() {
  std::cout << "Usage: pi pj pk di dj dk num_processors memsize\n"
            << "\n"
            << "Multiply a ('di'*'pi', 'dj'*'pj') matrix with\n"
            << "         a ('dj'*'pj', 'dk'*'pk') marrix\n"
            << "using the 3d matrix multiply algorithm.\n"
            << "\n"
            << "The multiply occurs over a virtual grid of\n"
            << "'pi'*'pj'*'pk' processors mapped to\n"
            << "'num_processors' physical processors\n"
            << "When compiling the memgraph, use `memsize` memory\n"
            << "per processor\n";
}

template <typename T>
void print_graphviz(T const& obj, string filename)
{
  std::ofstream f(filename);
  obj.print_graphviz(f);
  DOUT("printed " << filename);
}

int main(int argc, char** argv) {
  if(argc != 9) {
    usage();
    return 1;
  }

  int pi, pj, pk;
  uint64_t di, dj, dk;
  int num_processors;
  uint64_t mem_size;
  try {
    pi             = parse_with_ss<int>(     argv[1]);
    pj             = parse_with_ss<int>(     argv[2]);
    pk             = parse_with_ss<int>(     argv[3]);
    di             = parse_with_ss<uint64_t>(argv[4]);
    dj             = parse_with_ss<uint64_t>(argv[5]);
    dk             = parse_with_ss<uint64_t>(argv[6]);
    num_processors = parse_with_ss<int>(     argv[7]);
    mem_size       = parse_with_ss<uint64_t>(argv[8]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }

  graph_constructor_t g = three_dimensional_matrix_multiplication(
    pi,pj,pk, di,dj,dk, num_processors);

  print_graphviz(g.graph, "g.gv");

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  print_graphviz(taskgraph, "tg.gv");

  auto [_2, _3, memgraph] = memgraph_t::make(
    taskgraph,
    {},
    vector<uint64_t>(num_processors, mem_size));

  print_graphviz(memgraph, "mg.gv");
}


