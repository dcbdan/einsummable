#include "../src/matgraph/matgraph.h"
#include "../src/einsummable/reference.h"

void linear_regression()
{
  matgraph_t mgraph;

  uint64_t n = 10;
  uint64_t p = 5;
  uint64_t d = 1;

  int x    = mgraph.insert_input(n, p);
  int y    = mgraph.insert_input(n, d);
  int w    = mgraph.insert_input(p, d);
  int yhat = mgraph.insert_matmul_ss(x, w);

  // yhat - y
  int diff    = mgraph.insert_ewb(scalarop_t::make_sub(), yhat, y);
  // nd,nd->dd
  int sq_diff = mgraph.insert_matmul_ts(diff, diff);

  int grad = mgraph.backprop(sq_diff, {w})[0];

  std::cout << "x:    " << x    << std::endl;
  std::cout << "y:    " << y    << std::endl;
  std::cout << "w:    " << w    << std::endl;
  std::cout << "yhat: " << yhat << std::endl;
  std::cout << "grad: " << grad << std::endl;

  std::cout << std::endl;

  mgraph.print();

  std::cout << "-------------------------------" << std::endl;
  std::cout << std::endl;

  auto [graph, m_to_g] = mgraph.compile({yhat, sq_diff, grad});
  graph.print();

  map<int, buffer_t> input_buffers;
  for(auto const& m_inn: {x, y, w}) {
    auto const& inn = m_to_g.at(m_inn);
    auto const& node = graph.nodes[inn];
    uint64_t sz = product(node.op.out_shape());
    buffer_t buffer = std::make_shared<buffer_holder_t>(sz);
    buffer->random(-0.1, 0.1);
    input_buffers.insert({inn, buffer});
  }

  map<int, buffer_t> output_buffers =
    reference_compute_graph(graph, input_buffers);

  for(auto const& [_, buffer]: output_buffers) {
    std::cout << buffer << std::endl;
  }
}

void ff()
{
  matgraph_t graph;

  uint64_t n = 100;
  vector<uint64_t> ds {13, 14, 15, 16, 17};

  int input_data = graph.insert_input(n, ds[0]);
  int x = input_data;
  vector<int> weights;
  for(int i = 0; i != ds.size()-1; ++i) {
    int const& u = ds[i  ];
    int const& v = ds[i+1];

    weights.push_back(graph.insert_input(u, v));

    x = graph.insert_matmul_ss(x, weights.back());
    x = graph.insert_ew(scalarop_t::make_relu(), x);
  }
  int prediction = x;

  int errors = graph.insert_ewb(scalarop_t::make_mul(), prediction, prediction);

  // This is a goofy neural network without any nonlinearities.
  // The errors is an elementwise square of the prediction matrix,
  // so if gradient descent works, the prediction should converge to zero.

  vector<int> grad_weights = graph.backprop(errors, weights);

  std::cout << "input data:    " << input_data   << std::endl;
  std::cout << "prediction:    " << prediction   << std::endl;
  std::cout << "weights:       " << weights      << std::endl;
  std::cout << "grad weights:  " << grad_weights << std::endl;

  std::cout << std::endl;

  graph.print();
}

int main() {
  linear_regression();
}
