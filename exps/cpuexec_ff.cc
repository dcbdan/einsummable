#include "../src/matrixgraph/matrixgraph.h"
#include "../src/matrixgraph/ff.h"

#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/execute.h"
#include "../src/execution/cpu/mpi_class.h"

#include "../src/autoplace/autoplace.h"

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
  ff_sqdiff_t ff_info = ff_sqdiff_update(dn,dp,dd,dws,learning_rate);
  matrixgraph_t const& mgraph = ff_info.mgraph;

  int x             = ff_info.x;
  int y             = ff_info.y;
  int yhat          = ff_info.yhat;
  int sqdiff        = ff_info.sqdiff;
  vector<int> ws    = ff_info.wsinn;
  vector<int> wsnew = ff_info.wsout;

  auto settings = settings_t::default_settings();

  vector<int> outs = wsnew;
  outs.push_back(sqdiff);
  auto [graph, m_to_g] = mgraph.compile(outs);
  auto pls = single_loc_placements(graph);
  auto [inputs_g_to_t, outputs_g_to_t, taskgraph] = taskgraph_t::make(graph, pls);

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
  sqdiff = outputs_g_to_t.at(m_to_g.at(sqdiff))(0,0);

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
    buffer_x->random(-0.05, 0.05);
    buffers.insert({x, buffer_x});
  }
  // Set y
  {
    buffer_t buffer_y = std::make_shared<buffer_holder_t>(dn*dd);
    buffer_y->random(-0.05, 0.05);
    buffers.insert({y, buffer_y});
  }
  // Set init weights
  for(int i = 0; i != ws.size(); ++i) {
    int const& w = ws[i];
    auto [w_d0,w_d1] = ff_info.shape_wi(i);
    uint64_t w_sz = w_d0*w_d1;

    buffer_t buffer_w = std::make_shared<buffer_holder_t>(w_sz);
    buffer_w->random(-0.05, 0.05);
    buffers.insert({w, buffer_w});
  }

  gremlin_t gg;
  for(int i = 0; i != niter;  ++i) {
    execute(taskgraph, settings, mpi, buffers);

    float loss = buffers.at(sqdiff)->sum();
    //if(i % 75 == 0) {
      std::cout << "loss: " << loss << std::endl;
    //}

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

  ff(mpi, dn, dp, dd, dws, niter, 0.001);
}
