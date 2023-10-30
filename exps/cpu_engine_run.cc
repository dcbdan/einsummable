#include "../src/base/setup.h"

#include "../src/base/args.h"

#include "../src/einsummable/graph.h"

#include "../src/engine/communicator.h"
#include "../src/server/cpu/server.h"

#include "../src/autoplace/apart.h"
#include "../src/autoplace/loadbalanceplace.h"

void usage() {
  std::cout << "Setup usage: addr_zero is_client world_size memsize(GB)\n";
  std::cout << "Plus args for graphs\n";
}

graph_t build_graph(args_t& args);

vector<placement_t> autoplace(
  graph_t const& graph,
  int world_size,
  int num_threads_per);

int main(int argc, char** argv) {
  if(argc < 4) {
    usage();
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  int num_threads = std::max(1, int(std::thread::hardware_concurrency()));

  DOUT("world size:                      " << world_size);
  DOUT("memory allocated:                " << (mem_size/GB) << " GB");
  DOUT("number of threads in threadpool: " << num_threads)
  DOUT("dtype:                           " << default_dtype());

  cpu_mg_server_t server(communicator, mem_size, num_threads);

  if(is_rank_zero) {
    args_t args(argc-4, argv+4);
    args.set_default("pp", true);
    server.set_parallel_partialize(args.get<bool>("pp"));

    server.set_use_storage(false);

    // execute this
    graph_t graph = build_graph(args);
    vector<placement_t> pls = autoplace(graph, world_size, num_threads);

    args.set_default<int>("nrep", 1);
    int nrep = args.get<int>("nrep");
    for(int rep = 0; rep != nrep; ++rep) {

      // initialize input tensors and distribute across the cluster
      for(int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if(node.op.is_input()) {
          auto const& input = node.op.get_input();
          dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
          tensor.random("-0.01", "0.01");
          //tensor.ones();
          server.insert_tensor(gid, pls[gid], tensor);
        }
      }

      // execute
      server.execute_graph(graph, pls);

    }

    server.shutdown();
  } else {
    server.listen();
  }
}

vector<placement_t> autoplace(
  graph_t const& graph,
  int world_size,
  int num_threads_per)
{
  auto parts = autopartition_for_bytes(graph, world_size * num_threads_per);
  return load_balanced_placement(graph, parts, world_size, false);
}

using tensor_t     = graph_writer_t::tensor_t;
using full_dim_t   = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

graph_t build_matmul(uint64_t ni, uint64_t nj, uint64_t nk) {
  graph_writer_t writer;
  tensor_t lhs = writer.input({ni,nj});
  tensor_t rhs = writer.input({nj,nk});
  tensor_t out = writer.matmul(lhs, rhs).save();
  return writer.get_graph();
}

graph_t make_graph_ff(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});

  tensor_t w1 = writer.input({hidden, dim});
  tensor_t w2 = writer.input({dim, hidden});
  tensor_t w3 = writer.input({hidden, dim});

  tensor_t w1t = w1.transpose(0, 1);
  tensor_t w2t = w2.transpose(0, 1);
  tensor_t w3t = w3.transpose(0, 1);

  scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

  tensor_t a = writer.matmul(x, w1t);
  //tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

  tensor_t b = writer.matmul(x, w3t) ;

  tensor_t c = writer.mul(a, b);

  tensor_t out = writer.matmul(c, w2t);
  out.save_inplace();

  return writer.get_graph();
}

uint64_t compute_hidden(
  uint64_t dim,
  uint64_t multiple_of)
{
  uint64_t ret = 4 * dim;
  ret = uint64_t( (2.0 * ret) / 3.0 );
  ret = multiple_of * ( (ret + multiple_of - 1) / multiple_of );
  return ret;
}

graph_t build_graph(args_t& args)
{
  args.set_default("graph", "matmul");

  if(args.get<string>("graph") == "matmul") {
    args.set_default("ni", uint64_t(10000));
    args.set_default("nj", uint64_t(10000));
    args.set_default("nk", uint64_t(10000));
    return build_matmul(
      args.get<uint64_t>("ni"),
      args.get<uint64_t>("nj"),
      args.get<uint64_t>("nk"));
  } else if(args.get<string>("graph") == "ff") {
    args.set_default<uint64_t>("batch",       1   );
    args.set_default<uint64_t>("seqlen",      2048);
    args.set_default<uint64_t>("dim",         4096);
    args.set_default<uint64_t>("multiple_of", 256 );

    uint64_t batch       = args.get<uint64_t>("batch");
    uint64_t seqlen      = args.get<uint64_t>("seqlen");
    uint64_t dim         = args.get<uint64_t>("dim");
    uint64_t multiple_of = args.get<uint64_t>("multiple_of");

    uint64_t hidden = compute_hidden(dim, multiple_of);
    return make_graph_ff(batch * seqlen, hidden, dim) ;
  } else {
    throw std::runtime_error("invalid graph");
  }
}