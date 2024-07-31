#include "../src/base/args.h"
#include "../src/server/gpu/server.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/gwriter.h"
#include <cuda_profiler_api.h>

#include "../src/einsummable/super.h"

#include "modules.h"

#include <unordered_map>
using std::unordered_map;

//extern "C" void moncontrol(int);

void exp01(args_t& args, server_base_t* server) {
  args.set_default<uint64_t>("nrow", 10000);
  args.set_default<uint64_t>("ncol", 10000);
  uint64_t nrow = args.get<uint64_t>("nrow");
  uint64_t ncol = args.get<uint64_t>("ncol");

  auto num_gpu = args.get<int>("num_gpus");
  graph_writer_t writer;
  auto X = writer.input({10000, 10000});
  auto Y = writer.softmax_v1(X);

  X.save_inplace();
  Y.save_inplace();

  graph_t const& graph = writer.get_graph();
  
  vector<partition_t> parts;
  vector<placement_t> pls;
  for(auto const& node: graph.nodes) {
    pls.push_back(partition_t::singleton(node.op.shape()));
  }

  parts = apart01(graph, num_gpu, 1);
  pls = alocate01(graph, parts, num_gpu, 1000);

  {
    dbuffer_t dX = make_dbuffer(default_dtype(), nrow*ncol);
    dX.random("-0.0", "1.0");
    server->insert_tensor(X.get_id(), pls[X.get_id()], dX);
  }

  server->execute_graph(graph, pls);
}

void exp02(args_t& args, server_base_t* server) {
  set_default_dtype(dtype_t::f16);

  auto num_gpu = args.get<int>("num_gpus");

  args.set_default<uint64_t>("seqlen", 1000); 
  args.set_default<int>("n_attention", 4);

  uint64_t seqlen = args.get<uint64_t>("seqlen");
  int n_attention = args.get<int>("n_attention");

  DOUT("seqlen is " << seqlen);
  DOUT("n_attention is " << n_attention);

  model_args_t margs = model_args_t::llama_65B(1);
  graph_writer_t writer;
  auto full_freqs_cis = writer.input(
    { seqlen, uint64_div(margs.head_dim(), 2) },
    dtype_t::c64);

  auto x = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(seqlen),
    margs.full_dim()
  }));
  vector<attention_t> attentions;
  for(int i = 0; i != n_attention; ++i) {
    attentions.emplace_back(&writer, "attention.", margs, 0, std::nullopt);
    x = attentions.back().forward(x, full_freqs_cis, std::nullopt);
  }

  //graph_t const& graph_unfused = writer.get_graph();
  //auto [graph, unfused_inns_to_fused, _] = graph_unfused.fuse();
  //int full_freqs_cis_id = unfused_inns_to_fused.at(full_freqs_cis.get_id());
  
  graph_t const& graph = writer.get_graph();
  int full_freqs_cis_id = full_freqs_cis.get_id();

  vector<partition_t> parts;
  vector<placement_t> pls;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];

    pls.push_back(partition_t::singleton(node.op.shape()));

    if(gid == full_freqs_cis_id) {
      auto dtype = node.op.out_dtype();
      auto shape = node.op.shape();
      dbuffer_t d = transformer_t::form_full_freqs_cis(margs.dim, margs.n_heads, seqlen);
      server->insert_tensor(gid, pls.back(), d);
    } else if(node.op.is_input()) {
      auto dtype = node.op.out_dtype();
      auto shape = node.op.shape();
      dbuffer_t d = make_dbuffer(dtype, product(shape));
      d.random("-0.0001", "0.0001");
      server->insert_tensor(gid, pls.back(), d);
    }
  }
  // creating partitions and placements for multiple gpus
  args.set_default<int>("mv_bytes", 1000);
  int flops_per_byte_moved = args.get<int>("mv_bytes");
  parts = apart01(graph, num_gpu*1, 1);
  pls = alocate01(graph, parts, num_gpu, flops_per_byte_moved);
  // print all the placements
  // for(auto const& pl: pls) {
  //   DOUT("partition: " << pl.partition << " locs: " << pl.locations);
  // }
  
  server->execute_graph(graph, pls);
}

void exp03(args_t& args) {
  auto margs = model_args_t::llama_65B(1); 

  args.set_default<uint64_t>("seqlen", 1000); 
  args.set_default<int>("n_locs", 8);
  args.set_default<int>("n_config", 1);
  args.set_default<uint64_t>("discount_input_factor", 1);

  uint64_t seqlen = args.get<uint64_t>("seqlen");

  graph_writer_t writer;

  transformer_t model(&writer, margs, 0);

  auto x = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(seqlen),
    margs.full_dim()
  }));
  auto y = model.forward(x).save();

  for(auto [key, weight]: model.weight_map()) {
    weight.save_inplace();
  }
  for(auto [k,v]: model.get_new_kvs()) {
    k.save_inplace();
    v.save_inplace();
  }

  graph_t const& graph = writer.get_graph();

  map<int, set<int>> layer_to_gids; // For each layer, the graph nodes
  {
    for(int i = 0; i != model.layers.size(); ++i) {
      auto& layer = model.layers[i];
      for(auto const& [_, weight]: layer.weight_map()) {
        layer_to_gids[i].insert(weight.get_id());
        for(int gid = layer.mark1 + 1; gid <= layer.mark3; ++gid) {
          layer_to_gids[i].insert(gid);
        }
      }
    }
  
  //  map<int, int> w_inns; // For each weight, the layer
  //  for(int i = 0; i != model.layers.size(); ++i) {
  //    auto& layer = model.layers[i];
  //    for(auto const& [_, weight]: layer.weight_map()) {
  //      w_inns.insert({weight.get_id(), i});
  //      layer_to_gids[i].insert(weight.get_id());
  //    }
  //  }

  //  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
  //    auto const& node = graph.nodes[gid];
  //    for(int inn: node.get_inns_set()) {
  //      auto iter = w_inns.find(inn);
  //      if(iter != w_inns.end()) {
  //        auto const& which_layer = iter->second;
  //        layer_to_gids[which_layer].insert(gid);
  //      }
  //    }
  //  }
  }

  autoplace_config_t config = [&] {
    int n_locs = args.get<int>("n_locs");
    int n_config = args.get<int>("n_config");
    uint64_t discount_input_factor = args.get<uint64_t>("discount_input_factor");
    return autoplace_config_t::make_default01(
      n_locs, n_config, discount_input_factor);
  }();

  vector<partition_t> parts = apart01(
    graph, 
    config.n_compute(),
    config.max_branching(),
    config.discount_input_factor(),
    config.search_space());

  std::ofstream f("g.gv");
  graph.print_graphviz(f, parts);
  DOUT("printed g.gv");

  for(auto const& [which_layer, gids]: layer_to_gids) {
    string name = "g" + write_with_ss(which_layer) + ".gv";
    std::ofstream f(name);
    graph.print_subset_graphviz(f, gids, parts);
    DOUT("printed " << name);
  }
}

void exp02_fuse(args_t& args) {
  set_default_dtype(dtype_t::f16);

  auto num_gpu = args.get<int>("num_gpus");

  args.set_default<uint64_t>("seqlen", 1000); 
  args.set_default<int>("n_attention", 4);

  uint64_t seqlen = args.get<uint64_t>("seqlen");
  int n_attention = args.get<int>("n_attention");

  DOUT("seqlen is " << seqlen);
  DOUT("n_attention is " << n_attention);

  model_args_t margs = model_args_t::llama_65B(1);
  graph_writer_t writer;
  auto full_freqs_cis = writer.input(
    { seqlen, uint64_div(margs.head_dim(), 2) },
    dtype_t::c64);

  auto x = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(seqlen),
    margs.full_dim()
  }));
  vector<attention_t> attentions;
  for(int i = 0; i != n_attention; ++i) {
    attentions.emplace_back(&writer, "attention.", margs, 0, std::nullopt);
    x = attentions.back().forward(x, full_freqs_cis, std::nullopt);
  }

  graph_t const& graph = writer.get_graph();

  auto const& [g1, _0, _1] = graph.fuse(true, true);
  auto const& [g2, _2, _3] = graph.fuse(true, false);
  auto const& [g3, _4, _5] = graph.fuse(false, true);
  auto const& [g4, _6, _7] = graph.fuse(false, true);

  int cnt = 0;
  auto print_info = [&cnt](graph_t const& g) {
    string name = "g" + write_with_ss(cnt++) + ".gv";
    std::ofstream f(name);
    g.print_graphviz(f);
    DOUT("printed " << name);
  };

  print_info(graph);
  print_info(g1);
  print_info(g2);
  print_info(g3);
  print_info(g4);
}

void exp_super(int argc, char** argv) {
  args_t args(argc, argv);

  set_default_dtype(dtype_t::f16);

  args.set_default<uint64_t>("mem_size", 32);
  uint64_t GB = 1000lu * 1000lu * 1000lu;
  uint64_t mem_size = args.get<uint64_t>("mem_size") * GB;

  auto num_gpu = args.get<int>("num_gpus");

  args.set_default<uint64_t>("seqlen", 1000); 
  args.set_default<int>("n_attention", 4);

  uint64_t seqlen = args.get<uint64_t>("seqlen");
  int n_attention = args.get<int>("n_attention");

  DOUT("seqlen is " << seqlen);
  DOUT("n_attention is " << n_attention);

  model_args_t margs = model_args_t::llama_65B(1);
  graph_writer_t writer;
  auto full_freqs_cis = writer.input(
    { seqlen, uint64_div(margs.head_dim(), 2) },
    dtype_t::c64);

  auto x = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(seqlen),
    margs.full_dim()
  }));
  vector<attention_t> attentions;
  for(int i = 0; i != n_attention; ++i) {
    attentions.emplace_back(&writer, "attention.", margs, 0, std::nullopt);
    x = attentions.back().forward(x, full_freqs_cis, std::nullopt);
  }

  graph_t const& graph = writer.get_graph();
  vector<partition_t> parts;
  vector<placement_t> pls;

  // creating partitions and placements for multiple gpus
  args.set_default<int>("mv_bytes", 1000);
  int flops_per_byte_moved = args.get<int>("mv_bytes");
  parts = apart01(graph, num_gpu*1, 1);
  pls = alocate01(graph, parts, num_gpu, flops_per_byte_moved);

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);
  auto [_2, _3, memgraph] = memgraph_t::make_without_evict(
    taskgraph, 
    vector<uint64_t>(num_gpu, mem_size));

  super_graph_t super = create_super_graph(memgraph);

  for(auto const& node: super.nodes) {
    DOUT(node.ops.size());
  }

  vector<string> colors{
      "#61B292",
      "#AED09E",
      "#F1E8A7",
      "#A8896C",
      "#A8D8EA",
      "#AA96DA",
      "#FCBAD3",
      "#FFFFD2"};

  map<int, string> get_color;
  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    get_color.insert({mid, ""});
  }
  for(int sid = 0; sid != super.nodes.size(); ++sid) {
    //int loc = memgraph.nodes[super.nodes[sid].ops[0]].op.get_loc();
    for(auto const& mid: super.nodes[sid].ops) {
      get_color[mid] = colors[sid % colors.size()];
      //get_color[mid] = colors[loc];
    }
  }

  {
    std::ofstream f("mg.gv");
    memgraph.print_graphviz(f, get_color);
    DOUT("printed mg.gv");
  }

  {
    std::ofstream f("sg.gv");
    super.print_graphviz(f);
    DOUT("printed sg.gv");
  }

  DOUT("num memgraph nodes: " << memgraph.nodes.size());
  int num_input = 0;
  for(auto const& node: memgraph.nodes) {
    if(node.op.is_inputmem() || node.op.is_inputsto()) {
      num_input++;
    }
  }
  DOUT("num memgraph input nodes: " << num_input);
  DOUT("num super nodes: " << super.nodes.size());

  vector<int> cs;
  int num_bigger_than_one = 0;
  for(auto const& node: super.nodes) {
    cs.push_back(node.ops.size());
    if(node.ops.size() > 1) {
      num_bigger_than_one++;
    }
  }
  std::sort(cs.begin(), cs.end());
  DOUT("median count: " << cs[cs.size() / 2]);
  DOUT("num bigger htan one " << num_bigger_than_one);
  DOUT(cs);
}

//int main(int argc, char** argv) {
//  set_default_dtype(dtype_t::f16);
//
//  args_t args(argc, argv);
//
//  args.set_default<int>("num_gpus", 8);
//
//  args.set_default<uint64_t>("mem_size", 32);
//
//  exp02_fuse(args); 
//}

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f16);

  int world_size = 1;
  bool is_rank_zero = true;

  communicator_t communicator("0.0.0.0", is_rank_zero, world_size);

  args_t args(argc, argv);
  args.set_default<int>("num_gpus", 8);
  args.set_default<uint64_t>("mem_size", 32);
  //args.set_default<uint64_t>("storage_size", 1);

  int num_gpus = args.get<int>("num_gpus");
  if(num_gpus <= 0 || num_gpus > 8) {
    throw std::runtime_error("invalid number of gpus (hardcoded max: 8)");
  }

  uint64_t GB = 1000lu * 1000lu * 1000lu;
  uint64_t mem_size = args.get<uint64_t>("mem_size") * GB;
  //uint64_t storage_size = args.get<uint64_t>("storage_size") * GB;

  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i){
    buffer_sizes.push_back(mem_size);
  }

  auto gpu_server = new gpu_mg_server_t(communicator, buffer_sizes); // , storage_size);
  gpu_server->set_split_off_inputs(true);
  if(gpu_server->has_storage()) {
    throw std::runtime_error("should not be using storage");
  }

  auto num_iter = 2;
  for(int i = 0; i != num_iter; ++i) {
    DOUT("----- iteration " << i << " -----");
    exp02(args, gpu_server);
  }
  // exp01(args, server);

  return 0;
}
