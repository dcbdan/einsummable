#include "../src/base/args.h"
#include "../src/server/cpu/server.h"



taskgraph_t make_lots_of_sends_taskgraph(
  vector<uint64_t> tensor_sizes,
  int world_size,
  int niter)
{
  dtype_t dtype = default_dtype();

  taskgraph_t tg;

  for(uint64_t const& sz: tensor_sizes) {
    uint64_t nelem = sz / dtype_size(dtype);
    if(nelem == 0) {
      throw std::runtime_error("invalid size provided");
    }

    int loc = 0;
    int tid = tg.insert_input(loc, dtype, { nelem });
    for(int i = 0; i != niter; ++i) {
      bool save = i + 1 == niter;
      int new_loc = runif(world_size);
      if (loc == new_loc){
        new_loc = (new_loc + 1)%world_size;
      }
      tid = tg.insert_move(loc, new_loc, tid, save);
      loc = new_loc;
    }
  }

  return tg;
}

void main_rank_zero(server_base_t* server, args_t& args, int world_size)
{
  // TODO:
  // 1. create a graph
  // 2. insert the input tensors
  // 3. execute the graph
  taskgraph_t tg = make_lots_of_sends_taskgraph({10000,10000,10000},world_size,100000);
  {
    dtype_t dtype = default_dtype();
    map<int, tuple<int, buffer_t>> data;
    for(int tid = 0; tid != tg.nodes.size(); ++tid) {
      auto const& node = tg.nodes[tid];
      if(!node.op.is_input()) {
        continue;
      }
      uint64_t nelem = tg.nodes[tid].op.out_size() / dtype_size(dtype);
      dbuffer_t dbuffer = make_dbuffer(dtype, nelem);
      dbuffer.ones();
      data.insert({tid, {0, dbuffer.data}});
    }
    server->local_insert_tensors(data);
  }
  server->execute(tg, {});
}

int main(int argc, char** argv) {
  int expected_argc = 5;
  if(argc < expected_argc) {
    std::cout << "Need more arg.\n" << std::endl;
    return 1;
  }
  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;
  int num_threads = std::max(1, int(std::thread::hardware_concurrency()));
  int num_channels = 8;
  int num_channels_per_move = 1;
  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);
  int this_rank = communicator.get_this_rank();
  args_t args(argc-(expected_argc-1), argv+(expected_argc-1));
  cpu_mg_server_t server(
    communicator, mem_size, num_threads, num_channels_per_move);
  if(is_rank_zero) {
    main_rank_zero(&server, args, world_size);
    server.shutdown();
  } else {
    server.listen();
  }
}