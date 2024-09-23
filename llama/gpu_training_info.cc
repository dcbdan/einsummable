#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/gpu/server.h"

#include "../src/misc/checkpoint.h"
#include "../src/misc/update.h"

#include "../src/autoplace/autoplace.h"
#include <cuda_runtime_api.h>
#include <stdexcept>

string delimiter = "###END_OF_STRING###\n";

//./gpu_llama_train 7B gpus 1 memsize 32 use_cudagraph false split_off_inputs false max_n_layers 8 seq 512 storage 20
void usage() {
  std::cout << "Usage: addr_zero is_client world_size memsize "
               "base_data_file num_data_files (Args)\n"
               "Args:\n";
}

void print_part_pls(vector<partition_t> parts, vector<placement_t> full_pls){
  DOUT("printing placement:");
  for (int i = 0; i < full_pls.size(); i++) {
    DOUT("gid: " << i << " partition: " << full_pls[i].partition << " locations: " << full_pls[i].locations);
  }
}

void print_parts(vector<partition_t> parts){
  DOUT("printing partitions:");
  for (int i = 0; i < parts.size(); i++) {
    DOUT("gid: " << i << " num_parts: " << parts[i].num_parts());
  }
}

topology_t goofy_topology(){
  topology_t topo;

  topo.insert_link(0,-1,1); topo.insert_link(-1,0,1);
  topo.insert_link(1,-1,1); topo.insert_link(-1,1,1);
  topo.insert_link(2,-1,1); topo.insert_link(-1,2,1);
  topo.insert_link(3,-1,1); topo.insert_link(-1,3,1);
  topo.insert_link(4,-1,1); topo.insert_link(-1,4,1);
  topo.insert_link(5,-1,1); topo.insert_link(-1,5,1);
  topo.insert_link(6,-1,1); topo.insert_link(-1,6,1);
  topo.insert_link(7,-1,1); topo.insert_link(-1,7,1);

  topo.insert_link(0, 1, 4);
  topo.insert_link(0, 2, 8);
  topo.insert_link(0, 3, 4);
  topo.insert_link(0, 4, 8);
  topo.insert_path(0, -1, 5);
  topo.insert_path(0, -1, 6);
  topo.insert_path(0, -1, 7);

  topo.insert_link(1, 0, 4);
  topo.insert_link(1, 2, 4);
  topo.insert_link(1, 3, 8);
  topo.insert_path(1, -1, 4);
  topo.insert_link(1, 5, 8);
  topo.insert_path(1, -1, 6);
  topo.insert_path(1, -1, 7);

  topo.insert_link(2, 0, 8);
  topo.insert_link(2, 1, 4);
  topo.insert_link(2, 3, 8);
  topo.insert_path(2, -1, 4);
  topo.insert_path(2, -1, 5);
  topo.insert_link(2, 6, 4);
  topo.insert_path(2, -1, 7);

  topo.insert_link(3, 0, 4);
  topo.insert_link(3, 1, 8);
  topo.insert_link(3, 2, 8);
  topo.insert_path(3, -1, 4);
  topo.insert_path(3, -1, 5);
  topo.insert_path(3, -1, 6);
  topo.insert_link(3, 7, 4);

  topo.insert_link(4, 0, 8);
  topo.insert_path(4, -1, 1);
  topo.insert_path(4, -1, 2);
  topo.insert_path(4, -1, 3);
  topo.insert_link(4, 5, 4);
  topo.insert_link(4, 6, 8);
  topo.insert_link(4, 7, 4);

  topo.insert_path(5, -1, 0);
  topo.insert_link(5, 1, 8);
  topo.insert_path(5, -1, 2);
  topo.insert_path(5, -1, 3);
  topo.insert_link(5, 4, 4);
  topo.insert_link(5, 6, 4);
  topo.insert_link(5, 7, 8);

  topo.insert_path(6, -1, 0);
  topo.insert_path(6, -1, 1);
  topo.insert_link(6, 2, 4);
  topo.insert_path(6, -1, 3);
  topo.insert_link(6, 4, 8);
  topo.insert_link(6, 5, 4);
  topo.insert_link(6, 7, 8);

  topo.insert_path(7, -1, 0);
  topo.insert_path(7, -1, 1);
  topo.insert_path(7, -1, 2);
  topo.insert_link(7, 3, 4);
  topo.insert_link(7, 4, 4);
  topo.insert_link(7, 5, 8);
  topo.insert_link(7, 6, 8);

  return topo;
}

// void tokenizer_check(){
//   piper_t piper("./llama_tokenizer", "../tokenizer.model");
//   DOUT("pad_id  and vocab_size: " << parse_vector<int>(piper.read()));
//   string str = "This is a sentence";
//   int32_t sz = str.size();
//   char const* raw = reinterpret_cast<char const*>(&sz);
//   string msgsz(raw, raw + sizeof(sz));
//   piper.write(msgsz);
//   piper.write(str);
//   DOUT("parsing tokens");
//   vector<int> tokens = parse_vector<int>(piper.read());
//   DOUT(str);
//   DOUT(tokens);
// }

// int main(){
//   tokenizer_check();
// }

void main_rank_zero(
  tensor_reader2_t& model_loader,
  args_t& pargs,
  autoplace_config_t config);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f32);

  int expected_argc = 1;
  if(argc < expected_argc) {
    DOUT("Need to provide at least data file");
  }

  int num_data_files;

  string base_data_file(argv[1]);
  if(base_data_file == "7B") {
    num_data_files = 1;
  } else if(base_data_file == "13B") {
    num_data_files = 2;
  } else if(base_data_file == "30B") {
    num_data_files = 4;
  } else if(base_data_file == "65B") {
    num_data_files = 8;
  }
  base_data_file = "/home/zhimin/llama_files/es/" + base_data_file;

  args_t args(argc-1, argv+1);

  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;

  communicator_t communicator(addr_zero, is_rank_zero, world_size);
  int this_rank = communicator.get_this_rank();

  int num_gpus = args.get<int>("gpus");

  args.set_default<int>("computes", 1);
  int num_computes_per_loc = args.get<int>("computes");

  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i) {
    buffer_sizes.push_back(args.get<uint64_t>("memsize") * 1000lu * 1000lu * 1000lu);
  }

  uint64_t storage_size = args.get<uint64_t>("storage");
  storage_size *= 1000lu * 1000lu * 1000lu;

  args.set_default("use_cudagraph", false);
  bool use_cudagraph = args.get<bool>("use_cudagraph");

  args.set_default("split_off_inputs", false);

  DOUT("storage size:                    " << storage_size);

  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);

  tensor_reader2_t reader(
    num_gpus,
    base_data_file, num_data_files);

  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);

  main_rank_zero(reader, args, config);

  return 0;
}

struct graph_setup_t {
  model_args_t margs;
  graph_t full_graph;
  checkpoint_graphs_t checkpoint_graphs;
  updater_desc_t updater_desc;
  vector<tuple<int, int>> old_news;
  vector<tuple<int, fill_t>> init_fills;
  int embeddings_id;
  int predictions_id;
  int labels_id;
  int loss_id;
  int full_freqs_cis_id;
  vector<tuple<string, int>> model_weight_map;

  vector<uint64_t> get_shape(int id) const {
    return full_graph.out_shape(id);
  }
};

graph_setup_t make_graph(
  args_t& pargs,
  int llama_num_files,
  uint64_t batch_size)
{
  dtype_t dtype = default_dtype();

  pargs.set_default("seed", int(-1));
  int seed = pargs.get<int>("seed");
  if(seed >= 0) {
    set_seed(pargs.get<int>("seed"));
  }

  pargs.set_default<int>("lora_rank", 32);
  int lora_rank = pargs.get<int>("lora_rank");

  model_args_t margs = model_args_t::llama(llama_num_files, batch_size);

  pargs.set_default<int>("max_n_layers", -1);
  {
    int n_layers = pargs.get<int>("max_n_layers");
    DOUT("n_layers " << n_layers);
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  pargs.set_default<uint64_t>("seq", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("seq");
  DOUT("sequence length: " << margs.max_seq_len);

  graph_t graph;
  vector<int> checkpoints;
  vector<int> weight_ids;
  vector<int> grad_ids;
  set<int> forward_ids;

  int embeddings_id = -1;
  int predictions_id = -1;
  int labels_id = -1;
  int loss_id = -1;
  int full_freqs_cis_id = -1;
  vector<int> constant_ids;
  vector<tuple<string, int>> model_weight_map;
  {
    graph_writer_t writer;
    bool with_softmax_v3_scale = false;
    transformer_t model(&writer, margs, 0, lora_rank, with_softmax_v3_scale);

    tensor_t embeddings = writer.input(full_shape_t({
      full_dim_t::singleton(margs.batch_size),
      full_dim_t::singleton(margs.max_seq_len),
      margs.full_dim()
    }));

    // predictions: batch size, vocab size
    tensor_t predictions = model.forward(embeddings);
    tensor_t labels = writer.input(
      vector<uint64_t>{margs.batch_size, margs.vocab_size},
      dtype);

    // Compute the loss
    //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
    //   Loss = sum_n (l{n}) / N
    // Note, shift by c for numerical stability;
    //   where c{n} = max_c v{n,c}
    tensor_t loss;
    {
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
    loss.save_inplace();

    vector<tensor_t> ws;
    vector<tensor_t> cs;
    for(auto [name, tensor]: model.weight_map()) {
      model_weight_map.emplace_back(name, tensor.get_id());

      if(name.find("lora") != string::npos) {
        ws.push_back(tensor);
      } else {
        if(lora_rank) {
          // we're doing lora, so explicitly save all the weights
          tensor.save_inplace();
          // and add them as a constant
          cs.push_back(tensor);
        }
      }
    }

    {
      vector<int> fs = vector_iota<int>(writer.get_graph().nodes.size());
      forward_ids = set<int>(fs.begin(), fs.end());
    }

    vector<tensor_t> grads = writer.backprop(loss, ws);

    checkpoints = vector_from_each_method(model.checkpoints, int, get_id);
    weight_ids  = vector_from_each_method(ws, int, get_id);
    grad_ids    = vector_from_each_method(grads, int, get_id);

    embeddings_id = embeddings.get_id();
    predictions_id = predictions.get_id();
    labels_id = labels.get_id();
    loss_id = loss.get_id();

    full_freqs_cis_id = model.full_freqs_cis.get_id();
    model.full_freqs_cis.save_inplace();

    constant_ids = vector_from_each_method(cs, int, get_id);

    graph = std::move(writer.get_graph());
  }

  pargs.set_default<bool>("is_adamw", true);
  bool is_adamw = pargs.get<bool>("is_adamw");

  updater_desc_t::vanilla_t vanilla {};
  updater_desc_t::adamw_t adamw { .min_precision = dtype_t::f32 };
  updater_desc_t updater_desc = is_adamw ?
    updater_desc_t { dtype, adamw   }    :
    updater_desc_t { dtype, vanilla }    ;

  vector<tuple<int, int>> old_news;
  for(auto const& constant_id: constant_ids) {
    old_news.emplace_back(constant_id, constant_id);
  }

  old_news.emplace_back(full_freqs_cis_id, full_freqs_cis_id);

  vector<tuple<int, fill_t>> init_fills = update_weights(
    updater_desc, graph, old_news, weight_ids, grad_ids);

  // Note that update_weights may add input nodes, which must belong
  // to the forward graph
  for(auto const& [new_input_id, _]: init_fills) {
    forward_ids.insert(new_input_id);
  }

  // Make sure that every new is a save
  for(auto const& [old_id, new_id]: old_news) {
    graph.nodes[new_id].op.set_save(true);
  }

  checkpoint_graphs_t checkpoint_graphs(
    graph,
    checkpoints,
    forward_ids);

  return graph_setup_t {
    .margs = margs,
    .full_graph = graph,
    .checkpoint_graphs = checkpoint_graphs,
    .updater_desc = updater_desc,
    .old_news = old_news,
    .init_fills = init_fills,
    .embeddings_id = embeddings_id,
    .predictions_id = predictions_id,
    .labels_id = labels_id,
    .loss_id = loss_id,
    .full_freqs_cis_id = full_freqs_cis_id,
    .model_weight_map = model_weight_map
  };
}

void main_rank_zero(
  tensor_reader2_t& reader,
  args_t& pargs,
  autoplace_config_t config)
{
  dtype_t dtype = default_dtype();

  pargs.set_default("simplify_tg", false);
  set_tg_do_simplify(pargs.get<bool>("simplify_tg"));

  pargs.set_default("which", vector<int>());
  vector<int> which_data = pargs.get<vector<int>>("which");

  uint64_t batch_size;
  if(which_data.size() > 0) {
    batch_size = which_data.size();
  } else {
    pargs.set_default<uint64_t>("batch_size", 1);
    batch_size = pargs.get<uint64_t>("batch_size");
  }

  DOUT("batch_size: " << batch_size);

  // time to make the graph in ms
  auto start_graph = std::chrono::high_resolution_clock::now();
  graph_setup_t info = make_graph(pargs, reader.num_files(), batch_size);

  // print the full graph
  graph_t const& full_graph = info.full_graph;
  std::ofstream f("full_graph.gv");
  full_graph.print_graphviz(f);
  DOUT("printed full_graph.gv");

  auto end_graph = std::chrono::high_resolution_clock::now();
  DOUT("graph time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_graph - start_graph).count());

  // Used to figure out required shapes
  auto const& margs = info.margs;

  // Use graphs to get checkpoint_taskgraphs_t
  auto const& graphs = info.checkpoint_graphs;
  DLINEOUT("num of graphs: " << graphs.graphs.size());

  // updater_desc will be used to set learning rate scalar variables
  auto const& updater_desc = info.updater_desc;

  // these fills need to be added before executing
  auto const& init_fills = info.init_fills;

  DLINE;
  // used for the next remap
  auto const& old_news = info.old_news;
  vector<tuple<int, int>> next_iter_remap;
  next_iter_remap.reserve(old_news.size());
  for(auto const& [old_id, new_id]: old_news) {
    next_iter_remap.emplace_back(new_id, old_id);
  }
  DLINEOUT("number of locs " << config.n_locs());

  pargs.set_default("write_info", false);
  bool write_info = pargs.get<bool>("write_info");

  pargs.set_default("dir", "./");
  string save_directory = pargs.get<string>("dir"); 

  vector<partition_t> parts;
  vector<placement_t> full_pls;

  DOUT("creating partition and placement from scratch");
  parts = apart01(info.full_graph, config.n_locs(), 1, 1, parts_space_t::contraction);
  full_pls = alocate03(info.full_graph, parts, config.n_locs(), true);

  DOUT("writing decomp info (partition and placement)...");
  string part_path = save_directory + "decomp_part.txt";
  string pls_path = save_directory + "decomp_pls.txt";
  std::ofstream decomp_part_file(part_path);
  std::ofstream decomp_pls_file(pls_path);
  
  string parts_info = to_wire_partition_list(parts);
  string pls_info = to_wire_placement_list(full_pls);
  decomp_part_file << parts_info;
  decomp_pls_file << pls_info;
  DOUT("wrote decomp patition to " << part_path << " and decomp placement to " << pls_path);
}

