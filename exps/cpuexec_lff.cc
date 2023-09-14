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

void print_km_times(kernel_manager_t const& km) {
  double total = 0.0;
  for(auto const& [e, cn]: km.times) {
    auto const& [c,n] = cn;
    DOUT(e.str() << ": " << "[" << c << ", " << n << "]");
    total += c;
  }
  auto const& [c,n] = km.touch_times;
  DOUT("touch times: [" << c << ", " << n << "]");
  total += c;
  DOUT("total: " << total);
}

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

  tensor_t a = writer.matmul(x, w1t);
  //tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

  tensor_t b = writer.matmul(x, w3t) ;

  tensor_t c = writer.mul(a, b);

  tensor_t out = writer.matmul(c, w2t);
  out.save_inplace();

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

  a.save_inplace();

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

  print_km_times(km);

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

struct random_partition_t {
  partition_t operator()(vector<uint64_t> const& total_shape) {
    vector<partdim_t> partdims;
    for(uint64_t const& n: total_shape) {
      int p = vector_random(splits);
      if(p > n) {
        p = 1;
      }
      partdims.push_back(partdim_t::split(n, p));
    }
    return partition_t(partdims);
  }

  vector<int> splits;
};

tuple<vector<partition_t>, taskgraph_t>
random_partition_solve(graph_t const& g, int niter, cluster_settings_t const& sts)
{
  DOUT("random_partition_solve");
  random_partition_t rnd { .splits = {1,4,8,16} };//,16,32} };

  auto gen = [&]() {
    vector<partition_t> ret;
    ret.reserve(g.nodes.size());
    for(auto const& node: g.nodes) {
      ret.push_back(rnd(node.op.shape()));
    }
    return ret;
  };

  double best_cost = -1.0;
  vector<partition_t> ret_p;
  taskgraph_t ret_tg;
  for(int i = 0; i != niter; ++i) {
    if((i + 1) % 100 == 0) { DOUT(i + 1 << " / " << niter); }
    vector<partition_t> parts = gen();
    vector<placement_t> pls = load_balanced_placement(g, parts, 1, false);
    auto [_0, _1, tg] = taskgraph_t::make(g, pls);
    double cost = bytes_cost(tg, sts);
    if(best_cost < 0 || cost < best_cost) {
      best_cost = cost;
      ret_p  = parts;
      ret_tg = tg;
      DOUT(best_cost);
    }
  }
  return {ret_p, ret_tg};
}

struct adj_random_partition_t {
  adj_random_partition_t(graph_t const& graph) {
    int ngid = graph.nodes.size();
    info.resize(ngid);
    // detect all of the following type of edges:
    //   input->einsummable
    //   einsummable->formation
    //   formation->einsummable
    for(int dst_gid = 0; dst_gid != ngid; ++dst_gid) {
      auto const& dst_node = graph.nodes[dst_gid];
      auto const& inns = dst_node.inns;
      for(int which_inn = 0; which_inn != inns.size(); ++which_inn) {
        int const& src_gid = inns[which_inn];
        auto const& src_node = graph.nodes[src_gid];
        if(src_node.op.is_input() && dst_node.op.is_einsummable()) {
          auto const& e = dst_node.op.get_einsummable();
          info[src_gid].emplace_back(dst_gid, e.inns[which_inn]);
        } else if(src_node.op.is_einsummable() && dst_node.op.is_formation()) {
          int rank = dst_node.op.shape().size();
          vector<int> idxs(rank);
          std::iota(idxs.begin(), idxs.end(), 0);
          info[dst_gid].emplace_back(src_gid, idxs);
        } else if(src_node.op.is_formation() && dst_node.op.is_einsummable()) {
          auto const& e = dst_node.op.get_einsummable();
          info[src_gid].emplace_back(dst_gid, e.inns[which_inn]);
        }
      }
    }
  }

  optional<partition_t> operator()(int gid, vector<partition_t> const& ps) {
    auto const& is = info[gid];
    if(is.size() == 0) {
      return std::nullopt;
    }
    int which = runif(is.size());
    auto const& [other_gid, idxs] = is[which];
    vector<partdim_t> pds;
    for(int const& i: idxs) {
      pds.push_back(ps[other_gid].partdims[i]);
    }
    return partition_t(pds);
  }

  // gid -> [ gid, which index ]
  vector<vector<tuple<int, vector<int>>>> info;
};

tuple<vector<partition_t>, taskgraph_t>
random_step_partition_solve(
  graph_t const& g,
  int niter,
  cluster_settings_t const& sts,
  int max_distance)
{
  DOUT("random_step_partition_solve; max distance " << max_distance);
  random_partition_t rnd { .splits = {1,2,4, 8,12,16} }; // ,24,32,48,64} };
  adj_random_partition_t adj_rnd(g);

  vector<partition_t> current_ps;

  auto random_partition = [&](int gid) {
    if(runif(100) < 50) {
      auto maybe = adj_rnd(gid, current_ps);
      if(maybe) {
        return maybe.value();
      }
    }
    return rnd(g.nodes[gid].op.shape());
  };

  for(auto const& node: g.nodes) {
    current_ps.push_back(partition_t::singleton(node.op.shape()));
  }

  vector<partition_t> ret_p = current_ps;

  taskgraph_t ret_tg;
  double best_cost;
  {
    vector<placement_t> pls = load_balanced_placement(g, current_ps, 1, false);
    auto [_0, _1, tg] = taskgraph_t::make(g, pls);
    ret_tg = tg;
    best_cost = bytes_cost(tg, sts);
  }

  vector<int> all_choices(g.nodes.size());
  std::iota(all_choices.begin(), all_choices.end(), 0);

  vector<int> choices = all_choices;
  int distance = max_distance;
  for(int iter = 0; iter != niter; ++iter) {
    if((iter + 1) % 100 == 0) { DOUT(iter + 1 << " / " << niter); }

    if(distance == 0) {
      current_ps = ret_p;
      choices = all_choices;
      distance = max_distance;
    }

    int which = vector_random(choices);
    auto const& node = g.nodes[which];
    current_ps[which] = random_partition(which);

    vector<placement_t> pls = load_balanced_placement(g, current_ps, 1, false);
    auto [_0, _1, tg] = taskgraph_t::make(g, pls);
    double cost = bytes_cost(tg, sts);
    if(cost < best_cost) {
      ret_p = current_ps;
      ret_tg = tg;
      best_cost = cost;
      distance = max_distance;
      choices.resize(0);
      choices.push_back(which);
      for(auto const& out: node.outs) {
        choices.push_back(out);
      }
      set<int> inns(node.inns.begin(), node.inns.end());
      for(auto const& inn: inns) {
        choices.push_back(inn);
      }
      DOUT(best_cost);
    } else {
      distance--;
    }
  }

  return {ret_p, ret_tg};
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
  pargs.set_default<bool>    ("withstep",   true );

  uint64_t batch       = pargs.get<uint64_t>("batch");
  uint64_t seqlen      = pargs.get<uint64_t>("seqlen");
  uint64_t dim         = pargs.get<uint64_t>("dim");
  uint64_t multiple_of = pargs.get<uint64_t>("multiple_of");

  bool mm = pargs.get<bool>("mm");
  bool direct = pargs.get<bool>("direct");
  bool with_step = pargs.get<bool>("withstep");

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

  cluster_settings_t sts(1, nrunner);

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
    vector<partition_t> parts;
    if(with_step) {
      int max_distance = 10;
      auto [parts_, tg_] = random_step_partition_solve(g, 3000, sts, max_distance);
      parts = parts_;
      tg = tg_;
    } else {
      auto [parts_, tg_] = random_partition_solve(g, 3000, sts);
      parts = parts_;
      tg = tg_;
    }

    std::ofstream f("g_with_parts.gv");
    g.print_graphviz(f, parts);
    DOUT("printed g_with_parts.gv");


    //vector<partition_t> parts = {
    //  make_p(g.nodes[ 0].op.shape(), {2,1}),
    //  make_p(g.nodes[ 1].op.shape(), {4,1}),
    //  make_p(g.nodes[ 2].op.shape(), {2,4,1}),
    //  make_p(g.nodes[ 3].op.shape(), {3,3}),
    //};
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
    //vector<placement_t> pls = load_balanced_placement(g, parts, 1, false);
    //auto [iiss, saves, tg_] = taskgraph_t::make(g, pls);
    //DOUT("? " << iiss.size() << " " << saves.size() << " ?");
    //for(auto const& [save_gid, vtensor]: saves) {
    //  DOUT(save_gid << ": " << vtensor);
    //}
    //tg = tg_;
  }

  {
    string name = "tg_" + write_with_ss(nrunner) + ".gv";
    std::ofstream f(name);
    tg.print_graphviz(f);
    DOUT("printed " << name);
  }

  DOUT("tg cost: " << bytes_cost(tg, sts));

  vector<double> ts;
  for(int i = 0; i != nrep; ++i) {
    ts.push_back(execute_tg(tg, nrunner));
  }
  DOUT(ts);
  DOUT("execute with nrunner " << nrunner << ": " << vector_median(ts))
}

