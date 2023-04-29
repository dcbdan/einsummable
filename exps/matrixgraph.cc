#include "../src/matrixgraph/matrixgraph.h"
#include "../src/einsummable/reference.h"

void usage() {
  std::cout << "Usage: niter dn dp dd {dws}\n"
            << "\n"
            << "Train a feedforward neural network to predict\n"
            << "a random dn x dp data matrix\n"
            << "\n"
            << "niter: number of iterations\n"
            << "dn,dp: shape of input data matrix\n"
            << "dn,dd: shape of output data matrix\n"
            << "dws:   list of hidden dimensions\n"
            << "\n"
            << "This program is a reference implementation!\n";
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
  scalarop_t squared_difference =
    scalarop_t::from_string("power{2}[+[hole@0,*[hole@1,constant{-1}]]]");

  matrixgraph_t mgraph;

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

  int sq_diff = mgraph.insert_ewb(squared_difference, yhat, y);

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

    float loss = output_buffers.at(m_to_g.at(sq_diff))->sum();
    std::cout << loss << std::endl;

    for(int i = 0; i != ws.size(); ++i) {
      int const& w    = ws[i];
      int const& wnew = wsnew[i];
      input_buffers.at(m_to_g.at(w)) = output_buffers.at(m_to_g.at(wnew));
    }
  }
}

// X: n x p
// Y: n x d
void linear_regression(
  uint64_t dn, uint64_t dp, uint64_t dd,
  int niter,
  float learning_rate)
{
  return ff(dn, dp, dd, {}, niter, learning_rate);
}

void main01() {
  uint64_t dn = 100;
  uint64_t dp = 10;
  uint64_t dd = 1;
  int niter = 1000;
  float learning_rate = 0.3;
  linear_regression(dn, dp, dd, niter, learning_rate);
}

int main(int argc, char** argv) {
  if(argc < 5) {
    usage();
    return 1;
  }
  int niter;
  uint64_t dn, dp, dd;
  vector<uint64_t> dws;
  try {
    niter          = parse_with_ss<int>(     argv[1]);
    dn             = parse_with_ss<uint64_t>(argv[2]);
    dp             = parse_with_ss<uint64_t>(argv[3]);
    dd             = parse_with_ss<uint64_t>(argv[4]);
    for(int i = 5; i != argc; ++i) {
      dws.push_back( parse_with_ss<uint64_t>(argv[i]));
    }
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }

  float learning_rate = 0.1;
  ff(dn, dp, dd, dws, niter, learning_rate);
  return 0;
}
