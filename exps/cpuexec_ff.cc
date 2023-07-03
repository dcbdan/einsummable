#include "../src/matrixgraph/matrixgraph.h"
#include "../src/matrixgraph/ff.h"

#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/execute.h"
#include "../src/execution/cpu/mpi_class.h"

#include "../src/autoplace/autoplace.h"

#include <fstream>

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
  ff_sqdiff_t ff_info = ff_sqdiff_update(dn,dp,dd,dws,learning_rate,dtype);
  matrixgraph_t const& mgraph = ff_info.mgraph;

  set_default_dtype(dtype);

  int x             = ff_info.x;
  int y             = ff_info.y;
  int yhat          = ff_info.yhat;
  int sqdiff        = ff_info.sqdiff;
  vector<int> ws    = ff_info.wsinn;
  vector<int> wsnew = ff_info.wsout;

  auto settings = execute_taskgraph_settings_t::default_settings();

  vector<int> outs = wsnew;
  outs.push_back(sqdiff);
  auto [graph, m_to_g] = mgraph.compile(outs);
  auto pls = single_loc_placements(graph);
  auto [inputs_g_to_t, outputs_g_to_t, taskgraph] = taskgraph_t::make(graph, pls);

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("wrote tg.gv");
  }

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
    dbuffer_t buffer_x = make_dbuffer(dtype, dn*dp);
    buffer_x.random("-0.05", "0.05");
    buffers.insert({x, buffer_x.data});
  }
  // Set y
  {
    dbuffer_t buffer_y = make_dbuffer(dtype, dn*dd);
    buffer_y.random("-0.05", "0.05");
    buffers.insert({y, buffer_y.data});
  }

  // Set init weights
  for(int i = 0; i != ws.size(); ++i) {
    int const& w = ws[i];
    auto [w_d0,w_d1] = ff_info.shape_wi(i);
    uint64_t w_sz = w_d0*w_d1;

    dbuffer_t buffer_w = make_dbuffer(dtype, w_sz);
    buffer_w.random("-0.05", "0.05");
    buffers.insert({w, buffer_w.data});
  }

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  gremlin_t gg;
  for(int i = 0; i != niter;  ++i) {
    execute(taskgraph, settings, kernel_manager, &mpi, buffers);

    if(i % 10 == 0) {
      scalar_t loss = dbuffer_t(dtype, buffers.at(sqdiff)).sum();
      std::cout << "loss: " << loss.str() << std::endl;
    }

    for(int i = 0; i != ws.size(); ++i) {
      int const& w    = ws[i];
      int const& wnew = wsnew[i];
      buffers[w] = buffers.at(wnew);
    }
  }
}

void print_loop_kernel_info() {
  vector<string> ks = {
    "ite_>=[constant{f32|0},hole|f32@0,constant{f32|0},hole|f32@0]",
    "power{2}[+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]]",
    "*[constant{f32|2},+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]]",
    "*[hole|f32@0,hole|f32@1]",
    "*[ite_>=[constant{f32|0},hole|f32@0,constant{f32|0},constant{f32|1}],hole|f32@1]",
    "+[hole|f32@0,*[constant{f32|-1},*[constant{f32|0.01},hole|f32@1]]]",
    "ite_>=[constant{f16|0},hole|f16@0,constant{f16|0},hole|f16@0]",
    "power{2}[+[hole|f16@0,*[constant{f16|-1},hole|f16@1]]]",
    "*[constant{f16|2},+[hole|f16@0,*[constant{f16|-1},hole|f16@1]]]",
    "*[hole|f16@0,hole|f16@1]",
    "*[ite_>=[constant{f16|0},hole|f16@0,constant{f16|0},constant{f16|1}],hole|f16@1]",
    "+[hole|f16@0,*[constant{f16|-1},*[constant{f16|1000},hole|f16@1]]]",
    "+[hole|f16@0,*[constant{f16|-1},*[constant{f16|1000},hole|f16@1]]]",
    "ite_>=[constant{f64|0},hole|f64@0,constant{f64|0},hole|f64@0]",
    "power{2}[+[hole|f64@0,*[constant{f64|-1},hole|f64@1]]]",
    "*[constant{f64|2},+[hole|f64@0,*[constant{f64|-1},hole|f64@1]]]",
    "*[hole|f64@0,hole|f64@1]",
    "*[ite_>=[constant{f64|0},hole|f64@0,constant{f64|0},constant{f64|1}],hole|f64@1]",
    "+[hole|f64@0,*[constant{f64|-1},*[constant{f64|1000.1},hole|f64@1]]]",
    "+[hole|f64@0,*[constant{f64|-1},*[constant{f64|1000.1},hole|f64@1]]]"
  };

  auto to_type_str = [](dtype_t const& d) {
    if(d == dtype_t::f16) {
      return "float16_t";
    } else if(d == dtype_t::f32) {
      return "float";
    } else if(d == dtype_t::f64) {
      return "double";
    } else if(d == dtype_t::c64) {
      return "std::complex<float>";
    } else {
      throw std::runtime_error("should not reach");
    }
  };

  int nu = 0;
  int nb = 0;
  int nf = 0;
  for(auto const& s: ks) {
    scalarop_t f = parse_with_ss<scalarop_t>(s);
    auto const& [op_str, _] = f.to_cpp_bytes();
    int n_inn = f.num_inputs();
    std::cout << s << std::endl;
    std::cout << f.type_signature() << "|" << op_str << std::endl;
    if(n_inn == 1) {
      auto tout = to_type_str(f.out_dtype());
      auto tinn = to_type_str(f.inn_dtype(0).value());
      std::cout << "_unary_ew_loop(u" << (nu++) << ","
        << tout << "," << tinn << ","
        << op_str << ")" << std::endl;
    } else if(n_inn == 2) {
      auto tout = to_type_str(f.out_dtype());
      auto tlhs = to_type_str(f.inn_dtype(0).value());
      auto trhs = to_type_str(f.inn_dtype(1).value());
      std::cout << "_binary_ew_loop(b" << (nb++) << ","
        << tout << "," << tlhs << "," << trhs << ","
        << op_str << ")" << std::endl;
    } else {
      nf++;
    }
    std::cout << std::endl;
  }

  if(nf != 0) {
    throw std::runtime_error("COULD NOT PROCESS ALL");
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
