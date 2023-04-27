#include "../src/matgraph/matgraph.h"
#include "../src/einsummable/reference.h"

// X: n x p
// Y: n x d
void linear_regression(
  uint64_t dn, uint64_t dp, uint64_t dd,
  int niter,
  float learning_rate)
{
  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole@0"),
      scalarop_t::make_scale(learning_rate)
    }
  );

  matgraph_t mgraph;

  int x    = mgraph.insert_input(dn, dp);
  int y    = mgraph.insert_input(dn, dd);
  int w    = mgraph.insert_input(dp, dd);
  int yhat = mgraph.insert_matmul_ss(x, w);

  // yhat - y
  int diff    = mgraph.insert_ewb(scalarop_t::make_sub(), yhat, y);
  // nd,nd->dd
  int sq_diff = mgraph.insert_matmul_ts(diff, diff);

  int grad = mgraph.backprop(sq_diff, {w})[0];

  int wnew = mgraph.insert_ewb(gradupdate, w, grad);

  std::cout << "x:    " << x    << std::endl;
  std::cout << "y:    " << y    << std::endl;
  std::cout << "w:    " << w    << std::endl;
  std::cout << "yhat: " << yhat << std::endl;
  std::cout << "grad: " << grad << std::endl;
  std::cout << "wnew: " << wnew << std::endl;

  std::cout << std::endl;

  mgraph.print();

  std::cout << "-------------------------------" << std::endl;
  std::cout << std::endl;

  auto [graph, m_to_g] = mgraph.compile({sq_diff, wnew});
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
  //input_buffers.at(y)->zeros();

  map<int, buffer_t> output_buffers;
  for(int i = 0; i != niter;  ++i) {
    //std::cout << "Iteration " << (i+1) << " / " << niter << std::endl;

    output_buffers = reference_compute_graph(graph, input_buffers);

    std::cout << output_buffers.at(m_to_g.at(sq_diff)) << std::endl;

    input_buffers.at(m_to_g.at(w)) = output_buffers.at(m_to_g.at(wnew));
  }

  std::cout << output_buffers.at(m_to_g.at(wnew)) << std::endl;
}

void ff(
  uint64_t dn, uint64_t dp, uint64_t dd,
  vector<uint64_t> dws,
  int niter, float learning_rate)
{
  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole@0"),
      scalarop_t::make_scale(learning_rate)
    }
  );
  scalarop_t relu = scalarop_t::make_relu();

  matgraph_t mgraph;

  int x = mgraph.insert_input(dn, dp);
  int y = mgraph.insert_input(dn, dd);

  int yhat = x;
  vector<int> ws;
  {
    uint64_t dlast = dp;
    for(auto const& dw: dws) {
      ws.push_back(mgraph.insert_input(dlast, dw));
      yhat = mgraph.insert_matmul_ss(yhat, ws.back());
      yhat = mgraph.insert_ew(relu, yhat);
      dlast = dw;
    }
    ws.push_back(mgraph.insert_input(dlast, dd));
    yhat = mgraph.insert_matmul_ss(yhat, ws.back());
  }

  // yhat - y
  int diff    = mgraph.insert_ewb(scalarop_t::make_sub(), yhat, y);

  // nd,nd->dd
  int sq_diff = mgraph.insert_matmul_ts(diff, diff);

  vector<int> grads = mgraph.backprop(sq_diff, ws);

  vector<int> wsnew;
  for(int i = 0; i != ws.size(); ++i) {
    int const& g = grads[i];
    int const& w = ws[i];
    wsnew.push_back(mgraph.insert_ewb(gradupdate, w, g));
  }

  vector<int> outs = wsnew;
  outs.push_back(sq_diff);
  auto [graph, m_to_g] = mgraph.compile(outs);

  map<int, buffer_t> input_buffers;

  // Set x
  {
    buffer_t buffer_x = std::make_shared<buffer_holder_t>(dn*dp);
    buffer_x->random(-0.1, 0.1);
    input_buffers.insert({m_to_g.at(x), buffer_x});
  }
  // Set y
  {
    buffer_t buffer_y = std::make_shared<buffer_holder_t>(dn*dd);
    buffer_y->random(-0.1, 0.1);
    input_buffers.insert({m_to_g.at(y), buffer_y});
  }
  // Set init weights
  for(auto const& w: ws) {
    auto const& [d0,d1] = mgraph.shape(w);
    buffer_t buffer_w = std::make_shared<buffer_holder_t>(d0*d1);
    buffer_w->random(-0.3, 0.3);
    input_buffers.insert({m_to_g.at(w), buffer_w});
  }

  for(int i = 0; i != niter;  ++i) {
    //std::cout << "Iteration " << (i+1) << " / " << niter << std::endl;

    map<int, buffer_t> output_buffers =
      reference_compute_graph(graph, input_buffers);

    std::cout << output_buffers.at(m_to_g.at(sq_diff)) << std::endl;

    for(int i = 0; i != ws.size(); ++i) {
      int const& w    = ws[i];
      int const& wnew = wsnew[i];
      input_buffers.at(m_to_g.at(w)) = output_buffers.at(m_to_g.at(wnew));
    }
  }
}

void ff()
{
  matgraph_t graph;

  uint64_t n = 100;
  uint64_t d = 5;
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

int main01() {
  uint64_t dn = 100;
  uint64_t dp = 10;
  uint64_t dd = 1;
  int niter = 1000;
  float learning_rate = 0.3;
  linear_regression(dn, dp, dd, niter, learning_rate);
  return 0;
}

int main() {
  uint64_t dn = 20;
  uint64_t dp = 10;
  uint64_t dd = 1;
  int niter = 1000;
  vector<uint64_t> dws{20,20,20};
  float learning_rate = 0.3;
  ff(dn, dp, dd, dws, niter, learning_rate);
  return 0;
}
