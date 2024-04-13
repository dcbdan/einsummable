#include “../src/base/args.h”

#include “../src/server/cpu/server.h”

#include "../src/autoplace/autoplace.h"

graph_t make_graph01() {
  graph_writer_t writer;
  auto x = writer.input({10,20});
  auto y = writer.input({20,30});
  scalarop_t op = scalarop_t::make_relu();
  auto z = writer.matmul(x, y);
  auto w = writer.ew(op, {x,y});
  w.save_inplace();
  return writer.get_graph();
}

graph_t make_graph02() {
  graph_t graph;

  int x = graph.insert_input({10,20});
  int y = graph.insert_input({20,30});

  einsummable_t matmul = einsummable_t::from_matmul(10,20,30);
  int z = graph.insert_einsummable(matmul, {x, y});
  z = graph.insert_formation(z);		
  
  scalarop_t op = scalarop_t::make_relu();
  einsummable_t ew(
    {10,30},
    vector<vector<int>>{ { 0, 1 } },
    2,
    op);
  int w = graph.insert(ew, vector<int>{z});
  graph.nodes[w].op.set_save(true);

  return graph;
}

void main_rank_zero(server_base_t* server, args_t& args, int world_size)
{
  graph_t graph = make_graph01(); // make_graph02 returns the same graph

  int num_config_threads_per_machine = args.get<int>("config_threads");
  autoplace_config_t autoplace_config = autoplace_config_t::make_default01(
    world_size, num_config_threads_per_machine);
  vector<placement_t> pls = autoplace01(graph, autoplace_config);

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      dbuffer_t d = make_dbuffer(node.op.out_dtype(), product(node.op.out_shape()));
      d.random("-1.0", "1.0");
      server->insert_tensor(gid, pls[gid], d);
    }
  }

  server->execute(graph, pls);

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_save()) {
      DOUT(server->get_tensor_from_gid(gid);
    }
  }
}

int main(int argc, char** argv) {
  int expected_argc = 5;
  if(argc < expected_argc) {
    usage();
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

