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
  server_base_t* server,
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
  base_data_file = "/home/ubuntu/mnt/es/" + base_data_file;

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

  gpu_mg_server_t* gpu_server;
  if(storage_size > 0) {
    gpu_server = new gpu_mg_server_t(communicator, use_cudagraph, buffer_sizes, storage_size);
  } else {
    gpu_server = new gpu_mg_server_t(communicator, use_cudagraph, buffer_sizes);
  }

  std::unique_ptr<server_base_t> server = std::unique_ptr<server_base_t>(gpu_server);

  args.set_default("split_off_inputs", false);
  gpu_server->set_split_off_inputs(args.get<bool>("split_off_inputs"));

  DOUT("storage size:                    " << storage_size);
  DOUT("split_off_inputs:                " << gpu_server->split_off_inputs_);

  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);

  tensor_reader2_t reader(
    num_gpus,
    base_data_file, num_data_files);

  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);

  if(is_rank_zero) {
    main_rank_zero(server.get(), reader, args, config);
  } else {
    server->listen();
  }

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
  server_base_t* server,
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
  pargs.set_default("load_info", false);
  bool load_info = pargs.get<bool>("load_info");
  vector<partition_t> parts;
  vector<placement_t> full_pls;
  if (load_info){
    DOUT("loading decomp info (partition and placement)...");
    string load_path = pargs.get<string>("load_path");
    string part_path = load_path + "decomp_part.txt";
    string pls_path = load_path + "decomp_pls.txt";
    std::ifstream decomp_part_file(part_path);
    std::ifstream decomp_pls_file(pls_path);
    if (!decomp_part_file.good() || !decomp_pls_file.good()) {
      throw std::runtime_error("loading decomp info but info file not found");
    }
    std::stringstream buffer_parts;
    std::stringstream buffer_pls;

    // read all lines in decomp_info_file
    string parts_info = "";
    buffer_parts << decomp_part_file.rdbuf();
    parts_info = buffer_parts.str();

    string pls_info = "";
    buffer_pls << decomp_pls_file.rdbuf();
    pls_info = buffer_pls.str();

    parts = from_wire_partition_list(parts_info);
    full_pls = from_wire_placement_list(pls_info);

    DOUT("loaded partition and placement");
  } else {
    DOUT("creating partition and placement from scratch");
    auto num_computes = config.n_compute_per_loc();
    DOUT("num computes: " << num_computes);
    DOUT("num total compute: " << config.n_compute());
    parts = apart01(info.full_graph, config.n_compute(), 1, 1, parts_space_t::contraction);
    if (config.n_compute_per_loc() == 1){
      DOUT("using alocate 03");
      full_pls = alocate03(info.full_graph, parts, config.n_locs(), false);
    }
    else {
      uint64_t flops_per_byte_moved = 1000;
      DOUT("using alocate 01");
      full_pls = alocate01(info.full_graph, parts, config.n_locs(), flops_per_byte_moved);
    }
    
  }

  if (write_info){
    DOUT("writing decomp info (partition and placement)...");
    string part_path = "./decomp_part.txt";
    string pls_path = "./decomp_pls.txt";
    std::ofstream decomp_part_file(part_path);
    std::ofstream decomp_pls_file(pls_path);
    
    string parts_info = to_wire_partition_list(parts);
    string pls_info = to_wire_placement_list(full_pls);
    decomp_part_file << parts_info;
    decomp_pls_file << pls_info;
    DOUT("wrote decomp patition to " << part_path << " and decomp placement to " << pls_path);
  }
  // try alocate04
  // topology_t topo = goofy_topology();
  // vector<placement_t> full_pls = alocate04(info.full_graph, parts, config.n_locs(), topo);

  DLINE;
  tuple<map<int, relation_t>, map<int, relation_t>, taskgraph_t> full_tg_info;

  //{
  //  auto [inn_rels_, out_rels_, tg] = taskgraph_t::make(info.full_graph, full_pls);
  //  map<int, relation_t> inn_rels;
  //  for(auto const& [gid, tids]: inn_rels_) {
  //    inn_rels.insert({
  //      gid,
  //      relation_t {
  //        .dtype = info.full_graph.out_dtype(gid),
  //        .placement = full_pls[gid],
  //        .tids = tids
  //      }
  //    });
  //  }
  //  map<int, relation_t> out_rels;
  //  for(auto const& [gid, tids]: out_rels_) {
  //    out_rels.insert({
  //      gid,
  //      relation_t {
  //        .dtype = info.full_graph.out_dtype(gid),
  //        .placement = full_pls[gid],
  //        .tids = tids
  //      }
  //    });
  //  }
  //  full_tg_info = { inn_rels, out_rels, tg };
  //}
  {
    checkpoint_taskgraphs_t checkpoint_taskgraphs(graphs, full_pls);
    full_tg_info = create_barrier_taskgraph( 
      graphs.manager,
      checkpoint_taskgraphs);
  }

  tuple<map<int, relation_t>, map<int, relation_t>, taskgraph_t> no_barrier_tg_info;
  DLINE;

  /////////////////////////////////////////////////////////////////////////////
  // Read in all the tensors
  dbuffer_t embedding_matrix;
  {
    DLINE;
    map<int, relation_t> relations;
    map<int, tuple<int, buffer_t>> local_data;
    int current_tid = 0;

    auto read_into = [&](
      string const& name, 
      vector<uint64_t> const& shape,
      map<int, tuple<int, buffer_t>>& local)
    {
      auto [rel, ds] = reader(name, shape, current_tid);
      current_tid += rel.placement.num_parts();

      vector<int> const& locs = rel.placement.locations.get();
      vector<int> const& tids = rel.tids.get();

      if(locs.size() != ds.size() || tids.size() != ds.size()) {
        throw std::runtime_error("bwoah.");
      }

      for(int i = 0; i != ds.size(); ++i) {
        int const& tid = tids[i];
        int const& loc = locs[i];
        auto& d        = ds[i];
        local.insert({ tid, { loc, d } });
      }

      return rel;
    };

    auto read_into_local_data = [&](string const& name, vector<uint64_t> const& shape) {
      return read_into(name, shape, local_data);
    };

    auto insert_name = [&](int gid, string const& name) {
      relation_t rel = read_into_local_data(name, info.get_shape(gid));
      relations.insert({gid, rel});
    };

    DLINE;
    {
      vector<uint64_t> shape{ margs.vocab_size, margs.dim };
      map<int, tuple<int, buffer_t>> ds;
      relation_t rel = read_into("tok_embeddings.weight", shape, ds);
      server->local_insert_tensors(ds);
      embedding_matrix = server->get_tensor(rel);
      server->local_erase_tensors(rel.tids.get());
    }

    DLINE;
    for(auto const& [name, id]: info.model_weight_map) {
      if(name.find("lora") == string::npos) {
        insert_name(id, name);
      }
    }
    DLINE;

    // Tell the server about the relations and local data we've gathered
    server->local_insert_tensors(local_data);
    for(auto const& [gid, rel]: relations) {
      server->insert_gid_without_data(gid, rel);
    }
  }
  DLINE;

  for(auto const& [gid, fill]: init_fills) {
    if(!fill.is_constant()) {
      throw std::runtime_error("not implemented");
    }
    scalar_t const& value = fill.get_constant().value;
    server->insert_constant(gid, full_pls[gid], value);
  }

  DLINE;
  for(auto const& [name, id]: info.model_weight_map) {
    if(name.find("lora") != string::npos) {
      auto shape = info.get_shape(id);

      // For the lora, we have (X*L0)*L1 where L0 needs to be
      // initialized gaussiann and L1 needs to be initialized with zeros
      dbuffer_t dbuffer = make_dbuffer(dtype, product(shape));

      if(name.find("lora0") != string::npos) {
        dbuffer.rnorm();
        dbuffer.scale(scalar_t(dtype, write_with_ss(float(1e-3))));
      } else if(name.find("lora1") != string::npos) {
        dbuffer.zeros();
      } else {
        throw std::runtime_error("should not reach");
      }

      server->insert_tensor(id, shape, dbuffer);
    }
  }

  DLINE;
  server->insert_tensor(
    info.full_freqs_cis_id,
    info.get_shape(info.full_freqs_cis_id),
    transformer_t::form_position_interpolation_full_freqs_cis(margs, 2048));
  // TODO: this is how form_position_interpolation works?

  /////////////////////////////////////////////////////////////////////////////
  
  DLINE;
  pargs.set_default("tokenizer", "/home/ubuntu/mnt/es/tokenizer.model");
  pargs.set_default("dataset", "/home/ubuntu/mnt/es/redpaj_long_samples");
  pargs.set_default("learning_rate", 1e-9f);
  string tokenizer_file = pargs.get<string>("tokenizer");
  string dataset_file   = pargs.get<string>("dataset");

  // check if tokenizer_file and dataset_file are valid
  std::ifstream tokenizer_check(tokenizer_file);
  if(!tokenizer_check.good()) {
    throw std::runtime_error("could not open tokenizer file");
  }
  std::ifstream dataset_check(dataset_file);
  if(!dataset_check.good()) {
    throw std::runtime_error("could not open dataset file");
  }
  DLINE;

  dataset_reader_t data_loader(tokenizer_file, dataset_file);

  scalar_t _lr(dtype, write_with_ss(pargs.get<float>("learning_rate")));
  map<string, scalar_t> vars {
    { "beta1", scalar_t(dtype, "0.9") },
    { "beta2", scalar_t(dtype, "0.999") },
    { "eta", _lr },
    { "learning_rate", _lr },
  };

  DLINE;
  pargs.set_default<int>("niter", 1);
  int niter = pargs.get<int>("niter");
  DOUT("starting training")
  for(int iter = 1; iter != niter + 1; ++iter) {
    // Insert the actual (embeddings,label) data
    // Note that embeddings will need to be selected from the embedding matrix
    // and the labels will need to be one-hot encoded
    DOUT("Iter: " << iter)
    auto [data_tokens, label_tokens] = [&] {
      if(which_data.size() > 0) {
        vector<vector<int>> data_tokens;
        vector<int> label_tokens;
        for(auto const& which_datum: which_data) {
          auto [datum_tokens, label_token] =
            data_loader.datum(which_datum, margs.max_seq_len);
          data_tokens.push_back(datum_tokens);
          label_tokens.push_back(label_token);
        }
        return tuple<vector<vector<int>>, vector<int>>(data_tokens, label_tokens);
      }
      DOUT("random data");
      return data_loader.random_data(margs.batch_size, margs.max_seq_len);
    }();

    DLINE;
    server->insert_tensor(
      info.embeddings_id,
      info.get_shape(info.embeddings_id),
      data_loader.make_embedding(
        embedding_matrix,
        vector_flatten(data_tokens)));

    server->insert_tensor(
      info.labels_id,
      info.get_shape(info.labels_id),
      data_loader.one_hot_encode(dtype, label_tokens));

    DLINE;
    update_vars(updater_desc, iter, vars);

    DLINE;
    {
      auto const& [inn_rels, out_rels, taskgraph] = full_tg_info;
      DLINE;
      server->remap(inn_rels);
      DLINE;
      server->execute(taskgraph, out_rels, vars);
      DLINE;
    }
    DLINE;

    double loss_val = server->get_tensor_from_gid(info.loss_id).sum_to_f64();
    DOUT("loss: " << loss_val);
    if(std::isnan(loss_val) || std::isinf(loss_val)) {
      DOUT("loss is nan or inf");
      // throw std::runtime_error("loss is nan or inf");
    }
    if (niter > 1){
      server->remap_gids(next_iter_remap);
    }
  }
  DLINE;
}

