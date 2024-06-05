#include "../src/base/args.h"
#include "../src/server/gpu/server.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/gwriter.h"

void exp01(args_t& args, server_base_t* server) {
  uint64_t nrow = args.get<uint64_t>("nrow");
  uint64_t ncol = args.get<uint64_t>("ncol");

  graph_writer_t writer;
  auto X = writer.input({10000, 10000});
  auto Y = writer.softmax_v1(X);

  X.save_inplace();
  Y.save_inplace();

  graph_t const& graph = writer.get_graph();
  
  vector<placement_t> pls;
  for(auto const& node: graph.nodes) {
    pls.push_back(partition_t::singleton(node.op.shape()));
  }

  {
    dbuffer_t dX = make_dbuffer(default_dtype(), nrow*ncol);
    dX.random("-0.0", "1.0");
    server->insert_tensor(X.get_id(), pls[X.get_id()], dX);
  }

  server->execute_graph(graph, pls);
}

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f16);

  int world_size = 1;
  bool is_rank_zero = true;

  communicator_t communicator("0.0.0.0", is_rank_zero, world_size);

  args_t args(argc, argv);
  args.set_default<int>("num_gpus", 1);
  args.set_default<uint64_t>("mem_size", 16);
  args.set_default<uint64_t>("storage_size", 20);

  int num_gpus = args.get<int>("num_gpus");
  if(num_gpus <= 0 || num_gpus > 8) {
    throw std::runtime_error("invalid number of gpus (hardcoded max: 8)");
  }

  uint64_t GB = 1000lu * 1000lu * 1000lu;
  uint64_t mem_size = args.get<uint64_t>("mem_size") * GB;
  uint64_t storage_size = args.get<uint64_t>("storage_size") * GB;

  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i){
    buffer_sizes.push_back(mem_size);
  }

  auto gpu_ptr = new gpu_mg_server_t(communicator, buffer_sizes, storage_size);
  gpu_ptr->set_split_off_inputs(true);

  std::unique_ptr<server_base_t> server = std::unique_ptr<server_base_t>(gpu_ptr);

  exp01(args, gpu_ptr);

  return 0;
}
