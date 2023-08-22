#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/autoplace/autopart.h"

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

  auto fix = [](int64_t c) {
    return double(c) * 1e-9;
  };

  auto autoplace = [&](graph_t const& graph) {
    autopart_mcmc_t autopart(graph, pargs.threads, pargs.threads*2);
    for(int i = 0; i != pargs.steps; ++i) {
      if(i % 100000 == 0) {
        std::cout << fix(autopart.get_current_cost()) << " | "
          << fix(autopart.get_best_cost())
          << std::endl;
      }
      autopart.step(-1.0);
    }
    return autopart.get_best_placements();
  };

  model_args_t model_args = model_args_t::llama_7B(pargs.bsz);

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
