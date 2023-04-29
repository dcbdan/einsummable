#include "../src/matrixgraph/matrixgraph.h"

#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/execute.h"
#include "../src/execution/cpu/mpi_class.h"

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
            << "This program is not distributed!\n";
}

void ff(
  mpi_t& mpi,
  uint64_t dn, uint64_t dp, uint64_t dd,
  vector<uint64_t> dws,
  int niter, float learning_rate)
{
  auto settings = settings_t::default_settings();

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
  vector<uint64_t> ws_sizes;
  {
    uint64_t dlast = dp;
    for(auto const& dw: dws) {
      ws.push_back(mgraph.insert_input(dlast, dw));
      ws_sizes.push_back(dlast*dw);
      yhat = mgraph.insert_matmul_ss(yhat, ws.back());
      yhat = mgraph.insert_ew(relu, yhat);
      dlast = dw;
    }
    ws.push_back(mgraph.insert_input(dlast, dd));
    ws_sizes.push_back(dlast*dd);
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
  auto [inputs_g_to_t, outputs_g_to_t, taskgraph] = taskgraph_t::make(graph);

  //////////
  // REWRITE ALL IDS FROM MATRIX GRAPH TO TASKGRAPH
  x = inputs_g_to_t.at(m_to_g.at(x))(0,0);
  y = inputs_g_to_t.at(m_to_g.at(y))(0,0);
  for(int& w: ws) {
    w = inputs_g_to_t.at(m_to_g.at(w))(0,0);
  }

  for(int& w: wsnew) {
    w = outputs_g_to_t.at(m_to_g.at(w))(0,0);
  }
  sq_diff = outputs_g_to_t.at(m_to_g.at(sq_diff))(0,0);

  // NOW DON'T USE MATRIX GRAPH IDS
  //////////

  // explicitly save x and y so they don't
  // get erased from bufffers
  taskgraph.nodes[x].is_save = true;
  taskgraph.nodes[y].is_save = true;

  map<int, buffer_t> buffers;

  // Set x
  {
    buffer_t buffer_x = std::make_shared<buffer_holder_t>(dn*dp);
    buffer_x->random(-0.1, 0.1);
    buffers.insert({x, buffer_x});
  }
  // Set y
  {
    buffer_t buffer_y = std::make_shared<buffer_holder_t>(dn*dd);
    buffer_y->random(-0.1, 0.1);
    buffers.insert({y, buffer_y});
  }
  // Set init weights
  for(int i = 0; i != ws.size(); ++i) {
    int const& w = ws[i];
    int const& w_sz = ws_sizes[i];

    buffer_t buffer_w = std::make_shared<buffer_holder_t>(w_sz);
    buffer_w->random(-0.3, 0.3);
    buffers.insert({w, buffer_w});
  }

  for(int i = 0; i != niter;  ++i) {
    execute(taskgraph, settings, mpi, buffers);

    float loss = buffers.at(sq_diff)->sum();
    std::cout << "loss: " << loss << std::endl;

    for(int i = 0; i != ws.size(); ++i) {
      int const& w    = ws[i];
      int const& wnew = wsnew[i];
      buffers[w] = buffers.at(wnew);
    }
  }
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

  mpi_t mpi(argc, argv);
  if(mpi.world_size != 1) {
    throw std::runtime_error("This program is not distributed");
  }

  ff(mpi, dn, dp, dd, dws, niter, 0.1);
}
