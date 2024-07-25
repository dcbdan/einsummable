#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/gpu/server.h"

#include "../src/misc/checkpoint.h"
#include "../src/misc/update.h"

#include "../src/engine/repartition.h"

#include "../src/autoplace/autoplace.h"
#include <cuda_runtime_api.h>
#include <stdexcept>

void usage() {
  std::cout << "Usage: addr_zero is_client world_size memsize "
               "base_data_file num_data_files (Args)\n"
               "Args:\n";
}

template <typename T>
void print_graphviz(T const& obj, string filename)
{
  std::ofstream f(filename);
  obj.print_graphviz(f);
  DOUT("printed " << filename);
}

communicator_t gpu_mg_server_t::null_comm;

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
  gpu_mg_server_t* server,
  args_t& pargs,
  autoplace_config_t config,
  vector<uint64_t> buffer_sizes,
  int num_computes_per_loc,
  bool use_storage);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f32);

  int expected_argc = 1;
  if(argc < expected_argc) {
    DOUT("Need to provide at least data file");
  }

  // int num_data_files;

  // string base_data_file(argv[1]);
  // if(base_data_file == "7B") {
  //   num_data_files = 1;
  // } else if(base_data_file == "13B") {
  //   num_data_files = 2;
  // } else if(base_data_file == "30B") {
  //   num_data_files = 4;
  // } else if(base_data_file == "65B") {
  //   num_data_files = 8;
  // }
  // base_data_file = "/home/zhimin/mytmpfs/" + base_data_file;

  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;

  communicator_t communicator(addr_zero, is_rank_zero, world_size);
  int this_rank = communicator.get_this_rank();

  vector<uint64_t> buffer_sizes;
  // NOTE: 4 is hardcoded here since each anton has 4 gpus
  for (int i = 0; i < 4; ++i) {
    buffer_sizes.push_back(125lu * 100lu * 1000lu * 1000lu);
  }

  auto gpu_server = new gpu_mg_server_t(communicator, buffer_sizes, 10e10);
  std::unique_ptr<server_base_t> server = std::unique_ptr<server_base_t>(gpu_server);

  auto reader_process = [&](map<int, buffer_t> const& data_) {
    map<int, tuple<int, buffer_t>> data;
    for(auto const& [tid, buffer]: data_) {
      data.insert({tid, {this_rank, buffer}});
    }
    server->local_insert_tensors(data);
  };

  // tensor_reader_t reader(
  //   communicator,
  //   reader_process,
  //   this_rank, world_size,
  //   , num_data_files);

  // if(!is_rank_zero) {
  //   server->register_listen(
  //     reader.read_cmd(),
  //     [&]{ reader.listen_read(); });
  //   server->register_listen(
  //     reader.shutdown_cmd(),
  //     [&]{ reader.listen_shutdown(); });

  //   server->listen();

  //   return 0;
  // }

  args_t args(argc-1, argv+1);

  // args.set_default("use_storage", true);
  // gpu_server->set_use_storage(args.get<bool>("use_storage"));

  args.set_default("split_off_inputs", true);
  gpu_server->set_split_off_inputs(args.get<bool>("split_off_inputs"));

  args.set_default<int>("gpus", 4);
  args.set_default<int>("computes", 1);
  int num_gpus = args.get<int>("gpus");
  int num_computes_per_loc = args.get<int>("computes");
  
  DOUT("use_storage:                     " << gpu_server->has_storage());
  DOUT("split_off_inputs:                " << gpu_server->split_off_inputs_);

  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);

  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);

  main_rank_zero(gpu_server, args, config, buffer_sizes, num_computes_per_loc, gpu_server->has_storage());

  server->shutdown();

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

struct tensor_handler_t {
  graph_t full_graph;
  checkpoint_graphs_t checkpoint_graphs;
  map<int, int> gid_to_init_cg;
  checkpoint_taskgraphs_t checkpoint_taskgraphs;
  vector<memgraph_make_state_t> checkpoint_memgraphs;
  map<string, int> model_weight_map;
  gpu_mg_server_t* server;

  dbuffer_t get_tensor(string weight) {
    if (model_weight_map.find(weight) == model_weight_map.end()) {
      throw std::runtime_error("Requested weight does not exist");
    }
    return get_tensor(model_weight_map[weight]);
  }

  dbuffer_t get_tensor(int gid) {
    if (gid_to_init_cg.find(gid) == gid_to_init_cg.end()) {
      throw std::runtime_error("Unavailable gid entered to get_tensor");
    }
    int cg_gid = gid_to_init_cg[gid];
    relation_t& rel = checkpoint_taskgraphs.infos[0].init_rel.at(cg_gid);

    vector<uint64_t> shape = full_graph.out_shape(gid);
    dtype_t dtype = full_graph.out_dtype(gid);
    // relation_t rel = relation_t::make_singleton(dtype, shape, 0);
    // relation_t new_rel = relation_t::make_singleton(dtype, shape, 0);
    relation_t new_rel = rel.as_singleton(0);

    remap_relations_t remap;
    remap.insert(rel, new_rel);

    map<int, buffer_t> data;
    for (auto &tid : rel.tids.get()) {
      int mid = checkpoint_memgraphs[0].task_tensor_to_mem_node[tid];

      auto &node = checkpoint_memgraphs[0].memgraph.nodes[mid];

      buffer_t out;

      if (!node.op.is_inputmem() && !node.op.is_inputsto()) {
        throw std::runtime_error("Tensor to get does not correspond to inputmem or inputsto node");
      }
      if (node.op.is_inputmem()) {
        auto& input = node.op.get_inputmem();
        out = make_buffer(input.size);

        cudaError_t error = cudaMemcpy(out->raw(), increment_void_ptr(server->mems[input.loc], input.offset), input.size, cudaMemcpyDeviceToHost);
        if(error != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy failed");
        }
      } else {
        auto& input = node.op.get_inputsto();
        out = server->get_storage_buf(input.storage_id);
      }

      data[tid] = out;
    }

    repartition(server->null_comm, remap, data, nullptr);
    return dbuffer_t(dtype, data[0]);
  }

  void load_tensor(int gid, dbuffer_t d) {
    if (gid_to_init_cg.find(gid) == gid_to_init_cg.end()) {
      throw std::runtime_error("Unavailable gid entered to get_tensor");
    }
    int cg_gid = gid_to_init_cg[gid];
    relation_t& new_rel = checkpoint_taskgraphs.infos[0].init_rel.at(cg_gid);

    vector<uint64_t> shape = full_graph.out_shape(gid);
    dtype_t dtype = full_graph.out_dtype(gid);
    // relation_t rel = relation_t::make_singleton(dtype, shape, 0);
    
    relation_t rel = new_rel.as_singleton(0);
    remap_relations_t remap;
    remap.insert(rel, new_rel);

    map<int, buffer_t> data;
    data[0] = d.data;
    repartition(server->null_comm, remap, data, nullptr);

    for (auto &[tid, buf] : data) {
      int mid = checkpoint_memgraphs[0].task_tensor_to_mem_node[tid];

      auto &node = checkpoint_memgraphs[0].memgraph.nodes[mid];

      if (!node.op.is_inputmem() && !node.op.is_inputsto()) {
        throw std::runtime_error("Tensor to get does not correspond to inputmem or inputsto node");
      }
      if (node.op.is_inputmem()) {
        auto& input = node.op.get_inputmem();

        cudaError_t error = cudaMemcpy(increment_void_ptr(server->mems[input.loc], input.offset), buf->raw(), input.size, cudaMemcpyDeviceToHost);
        if(error != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy failed");
        }
      } else {
        auto& input = node.op.get_inputsto();
        server->set_storage_buf(buf, input.storage_id);
      }
    }
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
  DOUT("Lora rank: " << lora_rank);

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
    transformer_t model(&writer, margs, 0, lora_rank);

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
      // DOUT(name << ": " << std::to_string(tensor.get_id()));
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

    // set<int> tids;
    // for (auto& [tname, tid] : model_weight_map) {
    //   tids.insert(tid);
    //   DOUT("Tensor " << tname << " with tid " << tid << " is save node? " << graph.nodes[tid].op.is_save());
    // }

    // for (int i = 0; i < graph.nodes.size(); i++) {
    //   auto& node =  graph.nodes[i];
    //   if (node.op.is_save() && tids.count(i) == 0) {
    //     std::cout << i << " ";
    //   }
    // }
    // std::cout << std::endl;
    // print_graphviz(graph, "llama.gv");
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

  std::set<int> s(std::make_move_iterator(weight_ids.begin()),
              std::make_move_iterator(weight_ids.end()));
  // map<int, int> gid_remappings; // Things to include in mapping: updated lora to input lora, saved constants to input location, possibly data


  // Make sure that every new is a save
  for(auto const& [old_id, new_id]: old_news) {
    // if (s.count(old_id) > 0) {
    //   DOUT("Old id: " << old_id << " new id: " << new_id);
    // }
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
  gpu_mg_server_t* server,
  args_t& pargs,
  autoplace_config_t config,
  vector<uint64_t> buffer_sizes,
  int num_computes_per_loc,
  bool use_storage)
{
  dtype_t dtype = default_dtype();

  //
  pargs.set_default("simplify_tg", false);
  set_tg_do_simplify(pargs.get<bool>("simplify_tg"));
  //

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
  auto info = make_graph(pargs, 1, batch_size);
  auto end_graph = std::chrono::high_resolution_clock::now();
  DOUT("graph time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_graph - start_graph).count());
  // Used to figure out required shapes
  auto const& margs = info.margs;

  // Use graphs to get checkpoint_taskgraphs_t
  auto const& graphs = info.checkpoint_graphs;

  // updater_desc will be used to set learning rate scalar variables
  auto const& updater_desc = info.updater_desc;

  // these fills need to be added before executing
  auto const& init_fills = info.init_fills;

  // used for the next remap
  auto const& old_news = info.old_news;
  vector<tuple<int, int>> next_iter_remap;
  next_iter_remap.reserve(old_news.size());
  // DOUT("Old news: ");
  // for(auto const& [old_id, new_id]: old_news) {
  //   DOUT("Old id: " << old_id << ", new_id: " << new_id);
  //   next_iter_remap.emplace_back(new_id, old_id);
  // }

  auto start_pls = std::chrono::high_resolution_clock::now();
  vector<placement_t> full_pls = autoplace01(info.full_graph, config);
  for (auto &[old_gid, new_gid] : info.old_news) {
    full_pls[new_gid] = full_pls[old_gid];
  }
  checkpoint_taskgraphs_t taskgraphs(graphs, full_pls);

  // DOUT("First Taskgraph save relation: ")
  // for (auto &[key, value] : taskgraphs.infos[0].save_rel) {
  //   DOUT("Key: " << key << ", Values: ");
  //   auto vec = value.tids.get();
  //   for (int k = 0; k < vec.size(); k++) {
  //     DOUT(vec[k]);
  //   }
  // }

  // DOUT("\n\n Second remap: ")
  for (int i = 1; i < graphs.graphs.size(); i++) {
    for (auto &[old_gid, new_gid] : graphs.remaps[i]) {
      // DOUT("Old: " << std::to_string(old_gid) << ", New: " << std::to_string(new_gid));
      assert(taskgraphs.infos[i-1].save_rel.at(old_gid).tids.get().size() == taskgraphs.infos[i].init_rel.at(new_gid).tids.get().size());
    }
  }


  // DOUT("\n\nSecond taskgraph initial relation: ")
  // for (auto &[key, value] : taskgraphs.infos[1].init_rel) {
  //   DOUT("Key: " << key << ", Values: ");
  //   auto vec = value.tids.get();
  //   for (int k = 0; k < vec.size(); k++) {
  //     DOUT(vec[k]);
  //   }
  // }

  auto end_pls = std::chrono::high_resolution_clock::now();
  DOUT("placement time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_pls - start_pls).count() << " ms");


  auto start_mgmake = std::chrono::high_resolution_clock::now();

  vector<memgraph_make_state_t> states;
  states.reserve(taskgraphs.infos.size());
  
  // Start by constructing first memgraph
  auto &[init_rel, tg, save_rel] = taskgraphs.infos[0];
  vector<allocator_t> allocators;
  map<int, memstoloc_t> empty_map;
  for (auto &msize : buffer_sizes) {
    allocators.emplace_back(msize);
  }
  memgraph_make_state_t state(tg, {0, 0, 0, 0}, allocators, empty_map, 1, 0, true);

  for(int id = 0; id != tg.nodes.size(); ++id)
  {
    auto const& node = tg.nodes[id];
    if(node.op.is_input())
    {
      state.initialize_input(id);
    }
  }
  state.process(order_taskgraph(tg));
  print_graphviz(state.memgraph, "llamamg0.gv");
  states.push_back(state);
  


  for (int i = 1; i < taskgraphs.infos.size(); i++) {
    // DOUT("Creating memgraph " << i)
    auto &[_prev_init, _last_tg, save_rel] = taskgraphs.infos[i-1];
    auto &[init_rel, current_tg, _next_save] = taskgraphs.infos[i];
    memgraph_make_state_t prev_state = states[i-1];
    map<int, memstoloc_t> remappings;

    // DOUT("Building remappings for memgraph");

    for (auto &[old_gid, new_gid] : graphs.remaps[i]) {
      auto &old_tids = save_rel.at(old_gid).tids.get();
      auto &new_tids = init_rel.at(new_gid).tids.get();
      for (int which_tid = 0; which_tid < save_rel.at(old_gid).tids.get().size(); which_tid++) {
        // TODO Currently skipping shape of vtensors, assuming underlying vector is ordered similarly, check if ok?

        // DOUT("Getting tids to map between");
        int old_tid = old_tids[which_tid];
        int new_tid = new_tids[which_tid];


        // DOUT("Getting old mid");
        int old_mid =
        prev_state.task_tensor_to_mem_node[old_tid];
        // DOUT("Getting old node");
        auto &old_node = prev_state.memgraph.nodes[old_mid];
        // DOUT("Getting old memstoloc");
        memstoloc_t old_memstoloc = old_node.op.get_output_memstoloc();
        remappings[new_tid] = old_memstoloc;

        // int new_mid =
        //  states[i+1].task_node_to_mem_node[_which_node_t{new_tid}];
        // auto &new_node = states[i+1].memgraph.nodes[new_mid];
        // memloc_t new_memloc = new_node.op.get_inputmem().as_memloc();

        // if (!states[i+1].memgraph.nodes[new_mid].op.is_inputmem()) {
        //   DOUT("Second node is not inputmem, went wrong somewhere in code or in my understanding");
        // }


        // new_mem_mapping.emplace_back(old_memloc.offset, old_memloc.size, old_memloc.loc, new_memloc.offset, new_memloc.loc, old_tid);
      }
    }
    // DOUT("Building next memgraph");
    vector<allocator_t> allocators;
    for (auto &msize : buffer_sizes) {
      allocators.emplace_back(msize);
    }
    // DOUT("Initializing state");
    memgraph_make_state_t state(current_tg, {0, 0, 0, 0}, allocators, remappings, 1, 0, true);
    // DOUT("Processing nodes");
    auto ordering = order_taskgraph(current_tg);
    state.process(ordering);
    states.push_back(state);
  }

  
  auto end_mgmake = std::chrono::high_resolution_clock::now();
  DOUT("Memgraph make time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_mgmake - start_mgmake).count() << " ms");

  auto start_mgextend = std::chrono::high_resolution_clock::now();

  // Map gids that we want to move memory between to their respective gids in the first checkpoint graph
  map<int, int> final_gid_remaps;
  for (auto &[old_gid, new_gid] : graphs.remaps[graphs.remaps.size()-1]) {
    final_gid_remaps[new_gid] = old_gid;
  }

  map<int, int> initial_gid_remaps;
  for (auto &[old_gid, new_gid] : graphs.remaps[0]) {
    initial_gid_remaps[old_gid] = new_gid;
  }


  map<int, int> gid_remappings;
  for (auto &[old_gid, new_gid] : info.old_news) {
    gid_remappings[final_gid_remaps[new_gid]] = initial_gid_remaps[old_gid];
  }

  auto &[initial_rel, __last_tg, __save_rel] = taskgraphs.infos[0];
  auto &[__init_rel, __current_tg, final_rel] = taskgraphs.infos[taskgraphs.infos.size()-1];

  auto &init_state = states[0];
  auto &final_state = states[states.size()-1];

  vector<tuple<memstoloc_t, memstoloc_t, int>> new_mem_mapping;

  for (auto &[final_cg_gid, initial_cg_gid] : gid_remappings) {
    auto &final_tids = final_rel.at(final_cg_gid).tids.get();
    auto &init_tids = initial_rel.at(initial_cg_gid).tids.get();


    if (final_tids.size() != init_tids.size()) {
      throw std::runtime_error("Size of relations do not match up, make sure corresponding gids have the same placements");
    }

    for (int which_tid = 0; which_tid < final_rel.at(final_cg_gid).tids.get().size(); which_tid++) {

      int final_tid = final_tids[which_tid];
      int init_tid = init_tids[which_tid];

      int init_mid =
      init_state.task_tensor_to_mem_node[init_tid];
      auto &init_node = init_state.memgraph.nodes[init_mid];

      int final_mid =
      final_state.task_tensor_to_mem_node[final_tid];
      auto &final_node = final_state.memgraph.nodes[final_mid];

      memstoloc_t init_memstoloc = init_node.op.get_output_memstoloc();
      memstoloc_t final_memstoloc = final_node.op.get_output_memstoloc();

      new_mem_mapping.emplace_back(init_memstoloc, final_memstoloc, final_mid);
    }
  }

  states[states.size()-1].memgraph.set_prune(false); // Pruning causes an error, shouldn't be much of a performance hit

  final_state.move_tensors(new_mem_mapping);

  print_graphviz(states[states.size()-1].memgraph, "finalcpmg.gv");

  auto end_mgextend = std::chrono::high_resolution_clock::now();
  DOUT("Memgraph extend time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_mgextend - start_mgextend).count() << " ms");

  auto start_exec = std::chrono::high_resolution_clock::now();

  for (auto &state : states) {
    server->execute_memgraph(state.memgraph, false, {});
    vector<vector<std::array<int, 2>>> remaps;
    server->storage_remap_server(remaps);
  }

  auto end_exec = std::chrono::high_resolution_clock::now();
  DOUT("Memgraph execution time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_exec - start_exec).count() << " ms");

  map<int, int> gid_to_init_cg;
  for (auto &[from, to] : info.checkpoint_graphs.remaps[0]) {
    gid_to_init_cg[from] = to;
  }
  map<string, int> model_weights;
  for (auto &[name, gid] : info.model_weight_map) {
    model_weights[name] = gid;
  }

  tensor_handler_t tensors = tensor_handler_t {
    .full_graph = info.full_graph,
    .checkpoint_graphs = info.checkpoint_graphs,
    .gid_to_init_cg = gid_to_init_cg,
    .checkpoint_taskgraphs = taskgraphs,
    .model_weight_map = model_weights,
    .server = server
  };

  // ///////////////////////////////////////////////////////////////////////////
  // // Read in all the tensors
  // auto start_load = std::chrono::high_resolution_clock::now();
  // string register_cmd = server->get_registered_cmd();

  // dbuffer_t embedding_matrix;
  // vector<uint64_t> embedding_matrix_shape { margs.vocab_size, margs.dim };
  // {
  //   relation_t rel = model_loader(
  //     register_cmd,
  //     "tok_embeddings.weight",
  //     embedding_matrix_shape,
  //     server->get_max_tid() + 1);
  //   embedding_matrix = server->get_tensor(rel);
  // }

  // for(auto const& [name, id]: info.model_weight_map) {
  //   auto shape = info.get_shape(id);
  //   if(name.find("lora") != string::npos) {
  //     // For the lora, we have (X*L0)*L1 where L0 needs to be
  //     // initialized gaussiann and L1 needs to be initialized with zeros
  //     dbuffer_t dbuffer = make_dbuffer(dtype, product(shape));

  //     if(name.find("lora0") != string::npos) {
  //       dbuffer.rnorm();
  //       dbuffer.scale(scalar_t(dtype, write_with_ss(float(1e-3))));
  //     } else if(name.find("lora1") != string::npos) {
  //       dbuffer.zeros();
  //     } else {
  //       throw std::runtime_error("should not reach");
  //     }

  //     server->insert_tensor(id, shape, dbuffer);
  //   } else {
  //     int next_tid = server->get_max_tid() + 1;
  //     relation_t relation = model_loader(
  //       register_cmd, name, shape, next_tid);
  //     server->insert_gid_without_data(id, relation);
  //   }
  // }

  // model_loader.shutdown(register_cmd);

  // for(auto const& [gid, fill]: init_fills) {
  //   if(!fill.is_constant()) {
  //     throw std::runtime_error("not implemented");
  //   }
  //   scalar_t const& value = fill.get_constant().value;
  //   server->insert_constant(gid, full_pls[gid], value);
  // }

  // server->insert_tensor(
  //   info.full_freqs_cis_id,
  //   info.get_shape(info.full_freqs_cis_id),
  //   transformer_t::form_position_interpolation_full_freqs_cis(margs, 2048));
  // auto end_load = std::chrono::high_resolution_clock::now();
  // DOUT("load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count());
  // // TODO: this is how form_position_interpolation works?

  // /////////////////////////////////////////////////////////////////////////////
  
  // pargs.set_default("tokenizer", "../tokenizer.model");
  // pargs.set_default("dataset", "./redpaj_long_samples");
  // pargs.set_default("learning_rate", 1e-9f);
  // string tokenizer_file = pargs.get<string>("tokenizer");
  // string dataset_file   = pargs.get<string>("dataset");

  // // check if tokenizer_file and dataset_file are valid
  // std::ifstream tokenizer_check(tokenizer_file);
  // if(!tokenizer_check.good()) {
  //   throw std::runtime_error("could not open tokenizer file");
  // }
  // std::ifstream dataset_check(dataset_file);
  // if(!dataset_check.good()) {
  //   throw std::runtime_error("could not open dataset file");
  // }

  // dataset_reader_t data_loader(tokenizer_file, dataset_file);

  // scalar_t _lr(dtype, write_with_ss(pargs.get<float>("learning_rate")));
  // map<string, scalar_t> vars {
  //   { "beta1", scalar_t(dtype, "0.9") },
  //   { "beta2", scalar_t(dtype, "0.999") },
  //   { "eta", _lr },
  //   { "learning_rate", _lr },
  // };

  // pargs.set_default<int>("niter", 1);
  // int niter = pargs.get<int>("niter");
  // DOUT("starting training")
  // for(int iter = 1; iter != niter + 1; ++iter) {
  //   // Insert the actual (embeddings,label) data
  //   // Note that embeddings will need to be selected from the embedding matrix
  //   // and the labels will need to be one-hot encoded
  //   DOUT("Iter: " << iter)
  //   auto [data_tokens, label_tokens] = [&] {
  //     if(which_data.size() > 0) {
  //       vector<vector<int>> data_tokens;
  //       vector<int> label_tokens;
  //       // DOUT("which data: " << which_data);
  //       for(auto const& which_datum: which_data) {
  //         auto [datum_tokens, label_token] =
  //           data_loader.datum(which_datum, margs.max_seq_len);
  //         data_tokens.push_back(datum_tokens);
  //         label_tokens.push_back(label_token);
  //       }
  //       return tuple<vector<vector<int>>, vector<int>>(data_tokens, label_tokens);
  //     }
  //     DOUT("random data");
  //     return data_loader.random_data(margs.batch_size, margs.max_seq_len);
  //   }();

  //   server->insert_tensor(
  //     info.embeddings_id,
  //     info.get_shape(info.embeddings_id),
  //     data_loader.make_embedding(
  //       embedding_matrix,
  //       vector_flatten(data_tokens)));

  //   server->insert_tensor(
  //     info.labels_id,
  //     info.get_shape(info.labels_id),
  //     data_loader.one_hot_encode(dtype, label_tokens));

  //   update_vars(updater_desc, iter, vars);
  //   /////////////////////////////////
  //   // DANIEL MODIFICATIONS: Don't bother using all the checkpoint stuff,
  //   //                       just run 1 big taskgraph...
  //   DOUT("server executing graph");
  //   // server->execute_graph(info.full_graph, full_pls, vars);
  //   /////////////////////////////////
  //   for(int which = 0; which != taskgraphs.infos.size(); ++which) {
  //     // DOUT("server remapping");
  //     server->remap_gids(graphs.remaps[which]);
  //     auto const& [init_rels, taskgraph, save_rels] = taskgraphs.infos[which];
  //     server->remap(init_rels);
  //     DOUT("server executing");
  //     server->execute(taskgraph, save_rels, vars);
  //   }
  //   server->remap_gids(graphs.remaps.back());

  //   // double loss_val = server->get_tensor_from_gid(info.loss_id).sum_to_f64();
  //   // DOUT("loss: " << loss_val);
  //   // if(std::isnan(loss_val) || std::isinf(loss_val)) {
  //   //   throw std::runtime_error("loss is nan or inf");
  //   // }

  //   // server->remap_gids(next_iter_remap);
  // }
}
