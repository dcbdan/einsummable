#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

int main(int argc, char** argv) {
  args_t args(argc, argv);
  int this_rank = 0;

  // llama gpu parameters here
  args.set_default("parallel_partialize", false);
  args.set_default("use_storage", true);
  args.set_default("split_off_inputs", false);
  args.set_default<int>("gpus", 1);
  args.set_default<int>("computes", 1);
  args.set_default<int>("nseq", 512);
  args.set_default<int>("nbatch", 1);
  int num_gpus = args.get<int>("gpus");
  int num_computes_per_loc = args.get<int>("computes");
  int nseq = args.get<int>("nseq");
  int nbatch = args.get<int>("nbatch");

  // print parameters
  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);
  DOUT("nseq:                            " << nseq);
  DOUT("nbatch:                          " << nbatch);

  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  model_args_t model_args = model_args_t {
    .dim             = 4096, //was 4096
    .n_layers        = 1,
    .n_heads         = 32, //32
    .multiple_of     = 256, //256
    .norm_eps        = 1e-6,
    .batch_size      = 1,
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 32000,
  };

  builder_t builder = builder_t::make_first_token(model_args, uint64_t(512));
  graph_t graph = builder.graph;
  vector<int> inputs = graph.get_inputs();
  map<int, dbuffer_t> input_data;
  for (int input_id: inputs){
    dtype_t dtype = graph.out_dtype(input_id);
    auto shape = graph.out_shape(input_id);
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.1"));
    }
    input_data.insert({input_id, d});
  }

   
  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);


  vector<uint64_t> mem_sizes(1, 1000000000);

  auto [_2, _3, memgraph] = memgraph_t::make(taskgraph, {},  mem_sizes);
  std::ofstream f("mg.gv");
  memgraph.print_graphviz(f);
  DOUT("printed mg.gv");

  return 0;
}