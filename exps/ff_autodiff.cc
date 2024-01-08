#include "../src/base/setup.h"
#include "../src/base/args.h"

#include "../src/einsummable/gwriter.h"
#include "../src/server/cpu/server.h"

#include <fstream>

void usage() {
  std::cout << "Usage: addr_zero is_client world_size (args)\n"
               "Args:\n"
               "  niter\n"
               "  dn dp dd {dws}\n"
               "  learning_rate\n"
               "\n"
               "Train a feedforward neural network to predict\n"
               "a random dn x dp data matrix\n"
               "\n"
               "niter: number of iterations\n"
               "dn,dp: shape of input data matrix\n"
               "dn,dd: shape of output data matrix\n"
               "dws:   list of hidden dimensions\n";
}

struct info_t {
  int x_id;
  int y_id;
  vector<int> w_ids;
  int loss_id;
};

tuple<graph_t, info_t>
make_graph(
  uint64_t dn, uint64_t dp, uint64_t dd, vector<uint64_t> dws,
  float learning_rate)
{
  dtype_t dtype_before = default_dtype();
  set_default_dtype(dtype_t::f32);

  graph_writer_t gwriter;

  string ds = write_with_ss(dtype_t::f32);

  scalar_t lr = scalar_t(learning_rate);

  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(dtype_t::f32),
      scalarop_t::make_scale(lr)
    }
  );

  scalarop_t relu = scalarop_t::make_relu();

  scalarop_t squared_difference =
    scalarop_t::from_string(
      "power{2}[+[hole|"+ds+"@0,*[hole|"+ds+"@1,constant{"+ds+"|-1}]]]");

  using tensor = graph_writer_t::tensor_t;

  tensor x = gwriter.input({dn, dp});
  tensor y = gwriter.input({dn, dd});
  tensor yhat = x;
  vector<tensor> ws;
  {
    uint64_t dlast = dp;
    for(auto const& dw : dws) {
      ws.push_back(gwriter.input({dlast, dw}));
      yhat = gwriter.matmul(yhat, ws.back());
      yhat = gwriter.ew(relu, yhat);
      dlast = dw;
    }
    ws.push_back(gwriter.input({dlast, dd}));
    yhat = gwriter.matmul(yhat, ws.back());
  }

  tensor sqdiff  = gwriter.straight_bew(squared_difference, yhat, y);

  info_t ret {
    .x_id = x.get_id(),
    .y_id = y.get_id(),
    .w_ids = vector_from_each_method(ws, int, get_id),
    .loss_id = sqdiff.get_id()
  };

  return {gwriter.get_graph(), ret};
}

int main(int argc, char** argv) {
  if(argc < 3) {
    usage();
    return 1;
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  int num_threads = std::max(1, int(std::thread::hardware_concurrency()));
  int num_channels = 4;
  int num_channels_per_move = 1;

  args_t args(argc-3, argv+3);

  int niter = args.get<int>("niter");
  uint64_t dn = args.get<uint64_t>("dn");
  uint64_t dp = args.get<uint64_t>("dp");
  uint64_t dd = args.get<uint64_t>("dd");
  vector<uint64_t> dws = args.get<vector<uint64_t>>("dws");

  float learning_rate = args.get<float>("learning_rate");

  DOUT(niter << " " << dn << " " << dp << " " << dd << " " <<  dws << " " << learning_rate);

  auto [graph, info] = make_graph(dn, dp, dd, dws, learning_rate);

}
