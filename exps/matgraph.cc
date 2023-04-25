#include "../src/matgraph/matgraph.h"

void linear_regression()
{
  matgraph_t graph;

  uint64_t n = 100;
  uint64_t p = 20;
  uint64_t d = 1;

  int x    = graph.insert_input(n, p);
  int y    = graph.insert_input(n, d);
  int w    = graph.insert_input(p, d);
  int yhat = graph.insert_matmul_ss(x, w);

  // yhat - y
  int diff    = graph.insert_ewb(scalar_join_t::sub, yhat, y);
  // nd,nd->dd
  int sq_diff = graph.insert_matmul_ts(diff, diff);

  int grad = graph.backprop(sq_diff, {w})[0];

  std::cout << "x:    " << x    << std::endl;
  std::cout << "y:    " << y    << std::endl;
  std::cout << "w:    " << w    << std::endl;
  std::cout << "yhat: " << yhat << std::endl;
  std::cout << "grad: " << grad << std::endl;

  std::cout << std::endl;

  graph.print();
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
    x = graph.insert_ew(scalar_join_t::negate, x);
  }
  int prediction = x;

  int errors = graph.insert_ewb(scalar_join_t::mul, prediction, prediction);

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
