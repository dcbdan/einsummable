#include "../src/base/setup.h"
#include "../src/base/args.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/dbuffer.h"

#include "../src/execution/cpu/executetg.h"

#include "../src/autoplace/autopart.h"
#include "../src/autoplace/loadbalanceplace.h"

#include "../src/autoplace/bcost.h"

#include <fstream>

using tensor_t     = graph_writer_t::tensor_t;
using full_dim_t   = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

graph_t make_graph_ff_simple(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});

  tensor_t w1 = writer.input({hidden, dim});
  tensor_t w2 = writer.input({dim, hidden});
  tensor_t w3 = writer.input({hidden, dim});

  tensor_t w1t = w1.transpose(0, 1);
  tensor_t w2t = w2.transpose(0, 1);
  tensor_t w3t = w3.transpose(0, 1);

  scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

  tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

  tensor_t b = writer.matmul(x, w3t) ;

  tensor_t c = writer.mul(a, b);

  writer.matmul(c, w2t);

  return writer.get_graph();
}

graph_t make_graph_mm(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});

  tensor_t w1 = writer.input({hidden, dim});

  tensor_t w1t = w1.transpose(0, 1);

  tensor_t a = writer.matmul(x, w1t);

  return writer.get_graph();
}

buffer_t make_out(vector<uint64_t> const& shape) {
  dbuffer_t dbuffer = make_dbuffer(dtype_t::f32, product(shape));
  return dbuffer.data;
}

buffer_t make_data(vector<uint64_t> const& shape) {
  buffer_t ret = make_out(shape);
  dbuffer_t(dtype_t::f32, ret).random("-0.00001", "0.00001");
  return ret;
}

double execute_direct(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  kernel_manager_t km;

  buffer_t x = make_data({batch, dim});
  buffer_t w1 = make_data({hidden, dim});
  buffer_t w2 = make_data({dim, hidden});
  buffer_t w3 = make_data({hidden, dim});

  einsummable_t e1 = einsummable_t::from_matmul_st(batch,dim,hidden);

  einsummable_t ee({batch,hidden}, { {0,1} }, 2, scalarop_t::make_silu());

  einsummable_t e3 = einsummable_t::from_matmul_st(batch,dim,hidden);
  einsummable_t ea({batch, hidden}, { {0,1}, {0,1} }, 2, scalarop_t::make_add());

  einsummable_t e2 = einsummable_t::from_matmul_st(batch, hidden, dim);

  km.build(e1);
  km.build(ee);
  km.build(e3);
  km.build(ea);
  km.build(e2);

  auto start = clock_now();

  buffer_t z0 = make_out({batch,hidden});
  km(e1, z0->raw(), {x->raw(), w1->raw()});
  //km(ee, z0->raw(), {z0->raw()});

  buffer_t z1 = make_out({batch,hidden});
  km(e3, z1->raw(), {x->raw(), w3->raw()});

  km(ea, z1->raw(), {z0->raw(), z1->raw()});

  buffer_t z2 = make_out({batch,dim});
  km(e2, z2->raw(), {z1->raw(), w2->raw()});

  auto end = clock_now();

  std::chrono::duration<double, std::milli> duration = end - start;
  return duration.count();
}

double execute_direct_mm(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  kernel_manager_t km;

  buffer_t x = make_data({batch, dim});
  buffer_t w1 = make_data({hidden, dim});

  einsummable_t e1 = einsummable_t::from_matmul_st(batch,dim,hidden);

  km.build(e1);

  auto start = clock_now();

  buffer_t z0 = make_out({batch,hidden});
  km(e1, z0->raw(), {x->raw(), w1->raw()});

  auto end = clock_now();

  std::chrono::duration<double, std::milli> duration = end - start;
  return duration.count();
}

double execute_tg(
  taskgraph_t const& tg,
  int num_apply_runner)
{
  execute_taskgraph_settings_t settings {
    .num_apply_runner = num_apply_runner,
    .num_send_runner = 0,
    .num_recv_runner = 0
  };

  kernel_manager_t km;

  update_kernel_manager(km, tg);

  map<int, buffer_t> tensors;
  for(int id = 0; id != tg.nodes.size(); ++id) {
    auto const& node = tg.nodes[id];
    if(node.op.is_input()) {
      uint64_t nelem = node.op.get_input().size / dtype_size(default_dtype());
      tensors.insert({id, make_data({nelem})});
    }
  }

  auto start = clock_now();
  execute_taskgraph(tg, settings, km, nullptr, tensors);
  auto end = clock_now();

  std::chrono::duration<double, std::milli> duration = end - start;
  return duration.count();
}

uint64_t compute_hidden(
  uint64_t dim,
  uint64_t multiple_of)
{
  uint64_t ret = 4 * dim;
  ret = uint64_t( (2.0 * ret) / 3.0 );
  ret = multiple_of * ( (ret + multiple_of - 1) / multiple_of );
  return ret;
}

double vector_median(vector<double> xs) {
  int n = xs.size();
  if(n == 0) {
    throw std::runtime_error("no median of empty");
  }
  std::sort(xs.begin(), xs.end());
  return *(xs.begin() + (n/2));
}

partition_t make_p(vector<uint64_t> const& ds, vector<int> const& xs) {
  if(ds.size() != xs.size()) {
    throw std::runtime_error("make_p: incorect input sizes");
  }
  vector<partdim_t> pds;
  for(int i = 0; i != ds.size(); ++i) {
    uint64_t const& d = ds[i]; 
    int const& x = xs[i];
    pds.push_back(partdim_t::split(d, x));
  }
  return partition_t(pds);
}

int main(int argc, char** argv) {
  args_t pargs(argc, argv);

  pargs.set_default<uint64_t>("batch",       1   );
  pargs.set_default<uint64_t>("seqlen",      2048);
  pargs.set_default<uint64_t>("dim",         4096);
  pargs.set_default<uint64_t>("multiple_of", 256 );
  pargs.set_default<int>     ("nrep",        10  );
  pargs.set_default<int>     ("nrunner",     1   );
  pargs.set_default<bool>    ("mm",         true );
  pargs.set_default<bool>    ("direct",     false);

  uint64_t batch       = pargs.get<uint64_t>("batch");
  uint64_t seqlen      = pargs.get<uint64_t>("seqlen");
  uint64_t dim         = pargs.get<uint64_t>("dim");
  uint64_t multiple_of = pargs.get<uint64_t>("multiple_of");

  bool mm = pargs.get<bool>("mm");
  bool direct = pargs.get<bool>("direct");

  uint64_t hidden = compute_hidden(dim, multiple_of);

  int nrep = pargs.get<int>("nrep");
  int nrunner = pargs.get<int>("nrunner");

  graph_t g = mm                                      ?
    make_graph_mm(       batch * seqlen, hidden, dim) : 
    make_graph_ff_simple(batch * seqlen, hidden, dim) ;

  std::ofstream f("g.gv");
  g.print_graphviz(f);
  DOUT("printed g.gv");

  if(direct) {
    vector<double> ts;
    for(int i = 0; i != nrep; ++i) {
      if(mm) {
        ts.push_back(execute_direct_mm(batch * seqlen, hidden, dim));
      } else {
        ts.push_back(execute_direct(batch * seqlen, hidden, dim));
      }
    }
    DOUT(ts);
    DOUT("execute direct: " << vector_median(ts));
  }

  taskgraph_t tg;
  vector<partition_t> ps;
  if(nrunner == 1) {
    vector<placement_t> pls = g.make_singleton_placement();
    for(auto const& pl: pls) {
      ps.push_back(pl.partition);
    }
    auto [_0, _1, tg_] = taskgraph_t::make(g, pls);
    tg = tg_;
  } else {
    vector<partition_t> parts = {
      make_p(g.nodes[ 0].op.shape(), {1,1}),  
      make_p(g.nodes[ 1].op.shape(), {nrunner,1}),  
      make_p(g.nodes[ 2].op.shape(), {1,nrunner,1}),  
      make_p(g.nodes[ 3].op.shape(), {1,nrunner}),  
    };
    //vector<partition_t> parts = {
    //  make_p(g.nodes[ 0].op.shape(), {1,nrunner}),  
    //  make_p(g.nodes[ 1].op.shape(), {1,nrunner}),  
    //  make_p(g.nodes[ 2].op.shape(), {1,1,nrunner}),  
    //  make_p(g.nodes[ 3].op.shape(), {1,1}),  
    //};    
    //uint64_t min_sizing = 1;
    //auto parts = autopartition(g, min_sizing, nrunner);
    for(auto const& p: parts) {
      DOUT(p);
    }
    vector<placement_t> pls = load_balanced_placement(g, parts, 1, false);
    auto [_0, _1, tg_] = taskgraph_t::make(g, pls);
    tg = tg_;
  }

  {
    string name = "tg_" + write_with_ss(nrunner) + ".gv";
    std::ofstream f(name);
    tg.print_graphviz(f);
    DOUT("printed " << name);
  }

  cluster_settings_t sts(1, nrunner);
  DOUT("tg cost: " << bytes_cost(tg, sts));

  vector<double> ts;
  for(int i = 0; i != nrep; ++i) {
    ts.push_back(execute_tg(tg, nrunner));
  }
  DOUT(ts);
  DOUT("execute with nrunner " << nrunner << ": " << vector_median(ts))
}

