#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/autoplace/fsmcmc.h"
#include "../src/autoplace/rwmcmc.h"
#include "../src/autoplace/loadbalanceplace.h"
#include "../src/autoplace/autopart.h"

struct autoplace_t {
  vector<placement_t> operator()(graph_t const& graph) {
    int max_blocks = num_threads_per_node * nlocs * 2;

    relationwise_mcmc_t mcmc(
      graph, kernel_coster,
      nlocs, num_threads_per_node, max_blocks,
      equal_items_t<int>());

    DOUT("single thread cost " << mcmc.cost());

    {
      uint64_t min_sizing = 1;
      vector<partition_t> parts = autopartition(
        graph, min_sizing, nlocs * num_threads_per_node, mcmc.get_equal_gids());

      // this ignores the equal gids
      vector<placement_t> pls = load_balanced_placement(graph, parts, nlocs, false);

      mcmc.set_placements(pls);
    }

    DOUT("balanced cost " << mcmc.cost());

    for(int i = 0; i != num_steps; ++i) {
      if(i % 20000 == 0) {
        DOUT( i << " / " << num_steps << "    " << mcmc.cost() << " " << mcmc.get_best_cost() );
      }
      mcmc.step(beta);
    }

    DOUT(num_steps << " / " << num_steps << "   " << mcmc.get_best_cost() );
    return mcmc.get_best_placements();
  }

  kernel_coster_t kernel_coster;
  int nlocs;
  int num_threads_per_node;
  int num_steps;
  double beta;
};

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
      string("nlocs"),   argstrs, nlocs);

    read(int(1),
      string("threads"), argstrs, threads);

    read(int(0),
      string("steps"),   argstrs, steps);

    read(double(10000.0),
      string("beta"),    argstrs, beta);
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
};

int main(int argc, char** argv) {
  args_t pargs(argc, argv);

  auto kernel_coster = kernel_coster_t::for_cpu_cluster(pargs.nlocs);

  autoplace_t autoplace {
    .kernel_coster = kernel_coster,
    .nlocs = pargs.nlocs,
    .num_threads_per_node = pargs.threads,
    .num_steps = pargs.steps,
    .beta = pargs.beta
  };

  model_args_t model_args = model_args_t::llama_7B(pargs.bsz);
  model_args.n_layers = 2;

  builder_t builder = builder_t::make_first_token(model_args, pargs.seqlen, autoplace);

  int num_contraction_relation = 0;
  for(auto const& node: builder.graph.nodes) {
    if(node.op.is_einsummable()) {
      auto const& e = node.op.get_einsummable();
      if(e.is_contraction()) {
        num_contraction_relation++;
      }
    }
  }

  auto stats = builder.taskgraph.stats();
  DOUT("input bytes: " << toGBstr(stats.input_bytes));
  DOUT("form bytes:  " << toGBstr(stats.form_bytes));
  DOUT("move bytes:  " << toGBstr(stats.move_bytes));
  DOUT("save bytes:  " << toGBstr(stats.save_bytes));
  DOUT("touch bytes: " << toGBstr(stats.touch_bytes));
  DOUT("contraction rels:   " << num_contraction_relation);
  DOUT("contraction blocks: " << stats.contraction_blocks);
  DOUT("ew blocks:          " << stats.ew_blocks);
}
