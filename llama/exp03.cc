#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/autoplace/autopart.h"

#include <fstream>

double toGB(uint64_t n) {
  return n * 1e-9;
}
string toGBstr(uint64_t n) {
  return write_with_ss(toGB(n)) + " GB";
}

map<string, string> make_argstrs(int argc, char** argv) {
  if(argc % 2 != 0) {
    throw std::runtime_error("argstrs needs even number of args");
  }
  map<string, string> ret;
  for(int i = 0; i != argc; i += 2) {
    ret.insert({argv[i], argv[i+1]});
  }
  return ret;
}

struct args_t {
  args_t(int argc, char** argv)
  {
    map<string, string> argstrs = make_argstrs(argc-1, argv+1);

    read(uint64_t(1),
      string("bsz"),     argstrs, bsz);

    read(uint64_t(24),
      string("seqlen"),  argstrs, seqlen);

    read(int(1),
      string("threads"), argstrs, threads);

    read(int(0),
      string("steps"),   argstrs, steps);

    read(false,
      string("do_matmul"), argstrs, do_matmul);
  }

  template <typename T>
  static void read(
    T const& default_value,
    string const& key,
    map<string, string> const& argstrs,
    T& val)
  {
    auto iter = argstrs.find(key);
    if(iter == argstrs.end()) {
      val = default_value;
    } else {
      val = parse_with_ss<T>(iter->second);
    }
  }

  uint64_t bsz;
  uint64_t seqlen;
  int nlocs;
  int threads;
  int steps;
  double beta;
  bool do_matmul;
};

int main(int argc, char** argv) {
  args_t pargs(argc, argv);

  auto fix = [](int64_t c) {
    return double(c) * 1e-9;
  };

  auto autoplace_ = [&](graph_t const& graph) {
    autopart_mcmc_t mcmc(graph, pargs.threads, pargs.threads*2);
    for(int i = 0; i != pargs.steps; ++i) {
      if(i % 100000 == 0) {
        std::cout << fix(mcmc.get_current_cost()) << " | "
          << fix(mcmc.get_best_cost())
          << std::endl;
      }
      mcmc.step(-1.0);
    }
    auto& autopart = mcmc.autopart;
    auto& ginfos = autopart.ginfos;

    for(int gid = 0; gid != ginfos.size(); ++gid) {
      DOUT("ginfos[" << gid << "].partition = " << ginfos[gid].partition << ";");
    }

    return mcmc.get_best_placements();
  };
  auto autoplace = [&](graph_t const& graph) {
    autopart_t autopart(graph, pargs.threads);
    auto& ginfos = autopart.ginfos;

    for(auto& ginfo: ginfos) {
      auto& p = ginfo.partition;
      p[0] = 4;
      p[1] = 2;
      for(int i = 2; i != p.size(); ++i) {
        p[i] = 1;
      }
    }

    // set all inputs to an einsummable to whatever it is
    for(int gid = 0; gid != ginfos.size(); ++gid) {
      auto& ginfo = ginfos[gid];
      auto const& node = graph.nodes[gid];
      if(node.op.is_einsummable()) {
        auto const& e = node.op.get_einsummable();
        for(int i = 0; i != node.inns.size(); ++i) {
          int const& inn_gid = node.inns[i];
          auto const& inn_node = graph.nodes[inn_gid];
          if(inn_node.op.is_input()) {
            auto& inn_ginfo = ginfos[inn_gid];
            inn_ginfo.partition = e.get_input_from_join(ginfo.partition, i);
          }
        }
      }
    }

    std::ofstream f("gg.gv");
    autopart.print_graphviz(f);
    DOUT("printed gg.gv");

    return autopart.get_placements();
  };

  graph_t graph;
  taskgraph_t taskgraph;

  if(pargs.do_matmul) {
    uint64_t ni = 10000;
    uint64_t nj = 10000;
    uint64_t nk = 10000;
    int x = graph.insert_input({ni, nj});
    int y = graph.insert_input({nj, nk});
    int z = graph.insert_einsummable(
      einsummable_t::from_matmul(ni, nj, nk),
      {x,y});
    int w = graph.insert_formation(z);
    auto pls = autoplace(graph);
    for(auto const& pl: pls) {
      DOUT(pl.partition);
    }
    taskgraph = std::get<2>(taskgraph_t::make(graph, pls));
  } else {
    model_args_t model_args = model_args_t::llama_7B(pargs.bsz);

    model_args.n_layers = 2;

    builder_t builder = builder_t::make_first_token(model_args, pargs.seqlen, autoplace);
    graph = builder.graph;
    taskgraph = builder.taskgraph;
  }

  int num_contraction_relation = 0;
  for(auto const& node: graph.nodes) {
    if(node.op.is_einsummable()) {
      auto const& e = node.op.get_einsummable();
      if(e.is_contraction()) {
        num_contraction_relation++;
      }
    }
  }

  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("printed g.gv");
  }

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }

  auto stats = taskgraph.stats();
  DOUT("input bytes: " << toGBstr(stats.input_bytes));
  DOUT("form bytes:  " << toGBstr(stats.form_bytes));
  DOUT("move bytes:  " << toGBstr(stats.move_bytes));
  DOUT("save bytes:  " << toGBstr(stats.save_bytes));
  DOUT("touch bytes: " << toGBstr(stats.touch_bytes));
  DOUT("contraction rels:   " << num_contraction_relation);
  DOUT("contraction blocks: " << stats.contraction_blocks);
  DOUT("ew blocks:          " << stats.ew_blocks);
}


