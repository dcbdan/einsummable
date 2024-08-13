#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

void main_rank_zero_matmul(
  gpu_mg_server_t& server,
  args_t& args)
{
  int this_rank = 0;

  // llama gpu parameters here
  args.set_default<int>("gpus", 1);
  args.set_default<int>("computes", 1);
  args.set_default<uint64_t>("ni", 100);
  args.set_default<uint64_t>("nj", 100);
  args.set_default<uint64_t>("nk", 100);
  int num_gpus = args.get<int>("gpus");
  int num_computes_per_loc = args.get<int>("computes");
  uint64_t ni = args.get<uint64_t>("ni");
  uint64_t nj = args.get<uint64_t>("nj");
  uint64_t nk = args.get<uint64_t>("nk");

  // print parameters
  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);
  DOUT("ni:                              " << ni);
  DOUT("nj:                              " << nj);
  DOUT("nk:                              " << nk);

  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  //build the graph for feedforward only
  graph_writer_t writer;
  graph_t graph;
  tensor_t lhs1 = writer.input({ni,nj});
  tensor_t rhs1 = writer.input({nj,nk});
  tensor_t lhs2 = writer.input({ni,nj});
  tensor_t rhs2 = writer.input({nj,nk});
  tensor_t out1 = writer.matmul(lhs1, rhs1);
  tensor_t out2 = writer.matmul(lhs2, rhs2);
  tensor_t out = writer.matmul(out1, out2).save();


  graph = writer.get_graph();
  
  {
    std::cout << "g.gv" << std::endl;
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
  }

  //make random input values 
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
  std::cout << "Inputs: " << inputs << std::endl;

  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);

  std::cout << "Inputs: " << inputs << std::endl;
  for(auto const& id: graph.get_inputs()) {
    std::cout << "id: " << id << "  placements: " << placements[id].locations << std::endl;
    server.insert_tensor(id, placements[id], input_data.at(id));
  }
  server.execute_graph(graph, placements);
}


// ./gpu_llama 7B 1 max_n_layers n
int main(int argc, char** argv) {

  set_default_dtype(dtype_t::f32);

  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;

  if(is_rank_zero) {
    DOUT("world size:                      " << world_size);
  }

  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  int this_rank = communicator.get_this_rank();

  args_t args(argc, argv);

  vector<uint64_t> buffer_sizes;
  // NOTE: 4 is hardcoded here since each anton has 4 gpus
  for (int i = 0; i < 1; ++i) {
    buffer_sizes.push_back(1lu * 1000lu * 1000lu * 1000lu);
  }

  gpu_mg_server_t server(communicator, buffer_sizes);

  // auto reader_process = [&](map<int, buffer_t> const& data_) {
  //   map<int, tuple<int, buffer_t>> data;
  //   for(auto const& [tid, buffer]: data_) {
  //     data.insert({tid, {this_rank, buffer}});
  //   }
  //   server.local_insert_tensors(data);
  // };

  // tensor_reader_t reader(
  //   communicator,
  //   reader_process,
  //   this_rank, world_size,
  //   base_data_file, num_data_files);

  args.set_default("parallel_partialize", false);
  server.set_parallel_partialize(args.get<bool>("parallel_partialize"));

  args.set_default("use_storage", true);
  // server.set_use_storage(args.get<bool>("use_storage"));

  args.set_default("split_off_inputs", false);
  server.set_split_off_inputs(args.get<bool>("split_off_inputs"));

  // DOUT("parallel_partialize:             " << server.parallel_partialize_);
  DOUT("use_storage:                     " << server.use_storage_);
  DOUT("split_off_inputs:                " << server.split_off_inputs_);

  if(is_rank_zero) {
    main_rank_zero_matmul(server, args);
    
    server.shutdown();
  } else {
    server.listen();
  }
  return 0;
}