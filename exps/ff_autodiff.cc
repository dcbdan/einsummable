#include "../src/base/setup.h"
#include "../src/base/args.h"

#include "../src/einsummable/gwriter.h"
#include "../src/server/cpu/mg_server.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/trainer.h"

#include "../src/autoplace/autoplace.h"

#include <fstream>

void usage() {
  std::cout << "Usage: addr_zero is_client world_size memsize (Args)\n"
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
make_graph(uint64_t dn, uint64_t dp, uint64_t dd, vector<uint64_t> dws)
{
  dtype_t dtype_before = default_dtype();
  set_default_dtype(dtype_t::f32);

  graph_writer_t gwriter;

  string ds = write_with_ss(dtype_t::f32);

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

  // scale sqdiff so that the learning rate doesn't have to change
  // for different values of dn and dd
  tensor loss = sqdiff.scale(scalar_t(1/(float(1.0)*dn*dd)));

  info_t ret {
    .x_id = x.get_id(),
    .y_id = y.get_id(),
    .w_ids = vector_from_each_method(ws, int, get_id),
    .loss_id = loss.get_id()
  };

  return {gwriter.get_graph(), ret};
}

struct data_generator_t {
  data_generator_t(uint64_t dn_, uint64_t dp_, uint64_t dd_)
    : dn(dn_), dp(dp_), dd(dd_)
  {
    w = make_dbuffer(dtype_t::f32, dp*dd);
    w.random("-10.0", "10.0");
  }

  tuple<dbuffer_t, dbuffer_t>
  operator()() const
  {
    dbuffer_t x = make_dbuffer(dtype_t::f32, dn*dp);
    x.random("-1.0", "1.0");

    dbuffer_t y = make_dbuffer(dtype_t::f32, dn*dd);
    y.random("-0.01", "0.01");

    // np,pd->nd
    // ij,jk->ik
    matrix_multiply_update(
      dtype_t::f32,
      dn,dp,dd,
      false,false,
      y.raw(),
      x.raw(),
      w.raw(),
      false);

    return {x,y};
  }

  uint64_t dn;
  uint64_t dp;
  uint64_t dd;

  dbuffer_t w;
};

int main(int argc, char** argv) {
  if(argc < 4) {
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
  int num_channels = 4;
  int num_channels_per_move = 1;

  DOUT("n_locs " << world_size << " | num_threads_per_loc " << num_threads);

  communicator_t communicator(addr_zero, is_rank_zero, world_size);
  //cpu_mg_server_t server(communicator, mem_size, num_threads);
  cpu_tg_server_t server(communicator, mem_size, num_threads);

  if(!is_rank_zero) {
    server.listen();
    return 0;
  }

  // ^ initialize the server
  /////////////////
  // > create the graph + info

  args_t args(argc-4, argv+4);

  int niter = args.get<int>("niter");
  uint64_t dn = args.get<uint64_t>("dn");
  uint64_t dp = args.get<uint64_t>("dp");
  uint64_t dd = args.get<uint64_t>("dd");
  vector<uint64_t> dws = args.get<vector<uint64_t>>("dws");

  float learning_rate = args.get<float>("learning_rate");

  DOUT(niter << " " << dn << " " << dp << " " << dd << " " <<  dws << " " << learning_rate);

  auto [graph, info] = make_graph(dn, dp, dd, dws);

  /////////////////
  // > make a trainer object
  scalar_t lr = scalar_t(learning_rate);

  scalarop_t grad_update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(dtype_t::f32),
      scalarop_t::make_scale(lr)
    }
  );

  int num_config_threads_per_machine = num_threads;
  if(num_config_threads_per_machine == 12) {
    num_config_threads_per_machine = 8;
  }
  DOUT("NUM CONFIG THREADS " << num_config_threads_per_machine);
  autoplace_config_t config = autoplace_config_t::make_default02(
    world_size, num_config_threads_per_machine);
  auto f_autoplace = [&config](
    graph_t const& graph,
    map<int, placement_t> const& fixed_pls,
    vector<tuple<int,int>> const& equal_pls)
  {
    return autoplace02(graph, config, fixed_pls, equal_pls);
  };

  trainer_t trainer(
    &server,
    graph,
    info.loss_id,           // loss
    {info.loss_id},         // inspect
    {info.x_id, info.y_id}, // data
    {},                     // constants
    info.w_ids,             // weights
    grad_update,
    f_autoplace);
  // Here, we'll vary x at every iteration but
  // y will be held constant

  /////////////////
  // > initialize the weights
  for(int const& gid: info.w_ids) {
    auto shape = graph.out_shape(gid);
    dbuffer_t data = make_dbuffer(dtype_t::f32, product(shape));
    data.random("-0.1", "0.1");
    server.insert_tensor(gid, trainer.get_input_relation(gid), data);
  }

  /////////////////
  // > iterate
  data_generator_t gen(dn, dp, dd);

  int loss_so_far = 0.0;

  for(int iter = 1; iter != niter + 1; ++iter) {
    auto [xbuffer, ybuffer] = gen();
    server.insert_tensor(info.x_id, trainer.get_input_relation(info.x_id), xbuffer);
    server.insert_tensor(info.y_id, trainer.get_input_relation(info.y_id), ybuffer);
    trainer();
    double loss = server.get_tensor_from_gid(info.loss_id).sum_to_f64();
    loss_so_far += loss;
    if(iter % 100 == 0) {
      DOUT("avg loss at iter " << iter << ": " << (loss_so_far / 500.0));
      loss_so_far = 0.0;
      //for(int const& id: info.w_ids) {
      // DOUT("weight: " << server.get_tensor_from_gid(id));
      //}
    }
  }

  server.shutdown();
}


