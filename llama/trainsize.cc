#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/cpu/mg_server.h"
#include "../src/server/trainer.h"

#include "../src/autoplace/autoplace.h"

int main(int argc, char** argv) {
  args_t args(argc, argv);

  uint64_t mem_size = args.get<uint64_t>("mem_size");
  uint64_t GB = 1000000000;
  mem_size *= GB;

  int world_size  = args.get<int>("world_size");
  int num_threads = args.get<int>("num_threads");

  int lora_rank = args.get<int>("lora_rank");

  int num_model_files;
  string model_name = args.get<string>("model");
  if(model_name == "7B") {
    num_model_files = 1;
  } else if(model_name == "13B") {
    num_model_files = 2;
  } else if(model_name == "30B") {
    num_model_files = 4;
  } else if(model_name == "65B") {
    num_model_files = 8;
  } else {
    throw std::runtime_error("invalid model arg");
  }

  model_args_t margs = model_args_t::llama(
    num_model_files,
    args.get<uint64_t>("batch_size"));

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  margs.max_seq_len = args.get<uint64_t>("sequence_length");

  graph_writer_t writer;
  transformer_t model(&writer, margs, 0, lora_rank);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  tensor_t predictions = model.forward(embeddings);
  tensor_t labels = writer.input(
    vector<uint64_t>{margs.batch_size, margs.vocab_size},
    predictions.get_dtype());

  // Compute the loss
  //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
  //   Loss = sum_n (l{n}) / N
  // Note, shift by c for numerical stability;
  //   where c{n} = max_c v{n,c}
  tensor_t loss;
  {
    dtype_t dtype = predictions.get_dtype();
    tensor_t v = predictions;
    tensor_t c = writer.reduction("bv->b", castable_t::max, v);
    // v = v - c
    v = writer.ew("bv,b->bv", scalarop_t::make_sub(dtype), v, c);
    // ev = exp(v)
    tensor_t ev = writer.ew(scalarop_t::make_exp(dtype), v);
    // evsubset{b} = sum_v ev{b,v}*labels{b,v}
    tensor_t evsubset = writer.contraction("bv,bv->b", ev, labels);
    tensor_t evsum    = writer.reduction("bv->b", castable_t::add, ev);

    tensor_t lll = writer.ew(
      "b,b->b",
      scalarop_t::make_div(dtype),
      evsubset, evsum);

    lll = writer.ew(scalarop_t::make_log(dtype), lll);

    // (would like to use unsqueeze here but it is not implemented)

    double one_over_bsz = 1.0 / double(margs.batch_size);
    loss = lll.scale(scalar_t(dtype, write_with_ss(one_over_bsz)));
  }

  vector<int> trainer_weight_ids;
  vector<int> trainer_constant_ids;
  auto weight_map = model.weight_map();
  for(auto const& [name, tensor]: weight_map) {
    int id = tensor.get_id();
    if(name.find("lora") != string::npos) {
      trainer_weight_ids.push_back(id);
    } else if(name.find("norm") != string::npos) {
      trainer_weight_ids.push_back(id);
    } else {
      trainer_constant_ids.push_back(id);
    }
  }

  trainer_constant_ids.push_back(model.full_freqs_cis.get_id());

  DOUT("number of weight tensors: " << trainer_weight_ids.size());

  autoplace_config_t config =
    autoplace_config_t::make_default02(world_size, num_threads);

  auto f_autoplace = [&config](
    graph_t const& graph,
    map<int, placement_t> const& fixed_pls,
    vector<tuple<int,int>> const& equal_pls)
  {
    return autoplace02(graph, config, fixed_pls, equal_pls);
  };

  taskgraph_t taskgraph = trainer_t::dry_setup(
    writer.get_graph(),
    loss.get_id(),
    vector<int>{loss.get_id()},                        // inspect
    vector<int>{embeddings.get_id(), labels.get_id()}, // data
    trainer_constant_ids,
    trainer_weight_ids,
    f_autoplace,
    dtype_t::f32,
    update_type_t::adamw,
    false // don't make the gradients inspectable
  );

  bool success;
  try {
    auto [_0, _1, memgraph] = memgraph_t::make_without_evict(
      taskgraph,
      vector<uint64_t>(world_size, mem_size));
    DLINEOUT("memgraph size is " << memgraph.nodes.size());
    success = true;
  } catch(std::runtime_error const& error) {
    string msg = error.what();
    DOUT(msg);
    success = false;
  }

  std::cout << "world_size,mem_size,num_threads,lora_rank,model_name,"
               "batch_size,sequence_length,did_compile" << std::endl;
  std::cout << world_size << "," << (mem_size/GB) << "," << num_threads << ","
            << lora_rank << "," << model_name << ","
            << margs.batch_size << "," << margs.max_seq_len << ","
            << std::boolalpha << success
            << std::endl;

}

