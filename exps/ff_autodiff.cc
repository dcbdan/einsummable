#include "../src/einsummable/graph.h"
#include "../src/einsummable/einsummable.h"
#include <fstream>
#include "../src/base/setup.h"
#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/mpi_class.h"

void usage() {
  std::cout << "Usage: niter dn dp dd {dws} learning_rate\n"
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
  dtype_t dtype,
  mpi_t& mpi,
  uint64_t dn, uint64_t dp, uint64_t dd,
  vector<uint64_t> dws,
  int niter, float learning_rate) 
{

  if(dtype == dtype_t::c64) {
    throw std::runtime_error("ff_sqdiff_update does not support complex");
  }

  graph_writer_t gwriter; 

  dtype_t dtype_before = default_dtype();
  set_default_dtype(dtype);

  string ds = write_with_ss(dtype);

  scalar_t lr = scalar_t(learning_rate).convert(dtype);

  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole|"+ds+"@0"),
      scalarop_t::make_scale(lr)
    }
  );

  scalarop_t relu = scalarop_t::make_relu();

  scalarop_t squared_difference =
    scalarop_t::from_string(
      "power{2}[+[hole|"+ds+"@0,*[hole|"+ds+"@1,constant{"+ds+"|-1}]]]");

  using tensor = graph_writer_t::tensor_t;

  tensor x = gwriter.input({dn, dp});
  tensor y = gwriter.input({dn, dp});
  tensor yhat = x;
  vector<tensor> ws; 
  vector<uint64_t> ws_sizes;
  {
    uint64_t dlast = dp;
    for(auto const& dw : dws) {
      ws.push_back(gwriter.input({dlast, dw}));
      ws_sizes.push_back(dlast*dw);
      yhat = gwriter.matmul(y, ws.back());
      yhat = gwriter.ew(relu, yhat);
      dlast = dw;
    }
    ws.push_back(gwriter.input({dlast, dd}));
    ws_sizes.push_back(dlast*dd);
    yhat = gwriter.matmul(yhat, ws.back());
  }

  tensor softmax = gwriter.softmax(yhat);

  tensor sqdiff  = gwriter.straight_bew(squared_difference, softmax, y);

  vector<tensor> grads = gwriter.backprop(sqdiff, ws);

  vector<tensor> ws_new;
  for (int i = 0; i < ws.size(); ++i) {
    tensor const& g = grads[i];
    tensor const& w = ws[i];
    ws_new.push_back(gwriter.straight_bew(gradupdate, w, g));
  }

  auto settings = execute_taskgraph_settings_t::default_settings();
  graph_t graph = gwriter.get_graph();

  // We could create g_manager object instead of this 
  // We can insert some input data x,y ws...
  // manager.execute(graph)
  // Here is the new w_new vector<tuple<src,dst>> (gids only and manager will deal with tids)
  // Iterate over
  // Always save x y and yhat
  auto pls = graph.make_singleton_placement();
  auto [inputs_g_to_t, outputs_g_to_t, taskgraph] = taskgraph_t::make(graph, pls);

  {
    std::ofstream f("gw.gv");
    graph.print_graphviz(f);
    DOUT("wrote gw.gv");
  }
  
}



int main(int argc, char** argv) {
  //print_loop_kernel_info();
  //return 0;

  if(argc < 5) {
    usage();
    return 1;
  }
  int niter;
  uint64_t dn, dp, dd;
  vector<uint64_t> dws;
  float learning_rate;
  try {
    niter          = parse_with_ss<int>(     argv[1]);
    dn             = parse_with_ss<uint64_t>(argv[2]);
    dp             = parse_with_ss<uint64_t>(argv[3]);
    dd             = parse_with_ss<uint64_t>(argv[4]);
    for(int i = 5; i != argc-1; ++i) {
      dws.push_back( parse_with_ss<uint64_t>(argv[i]));
    }
    learning_rate = parse_with_ss<float>(argv[argc-1]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }

  mpi_t mpi(argc, argv);
  if(mpi.world_size != 1) {
    throw std::runtime_error("This program is not distributed");
  }

  ff(dtype_t::f64, mpi, dn, dp, dd, dws, niter, learning_rate);
}