#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"

#include "../src/misc/update.h"

#include "../src/autoplace/autoplace.h"

#include "../src/einsummable/gwriter.h"

using tensor_t = graph_writer_t::tensor_t;

void main_rank_zero(
  int world_size,
  server_base_t* server,
  args_t& args);

int main(int argc, char** argv) {
  {
    set_default_dtype(dtype_t::f64);
    cpu_kernel_executor_t k;
    auto e = einsummable_t::from_matmul(100, 100, 100);
    dbuffer_t X = make_dbuffer(dtype_t::f64, 100*100);
    X.random();

    dbuffer_t Y = make_dbuffer(dtype_t::f64, 100*100);
    Y.random();

    dbuffer_t Z = make_dbuffer(dtype_t::f64, 100*100);

    k.build(e);
    k(e, Z.raw(), vector<void const*>{X.raw(), Y.raw()},
      optional<buffer_t>(std::nullopt));
  }

  int expected_argc = 8;
  if(argc < expected_argc) {
    return 1;
  }

  set_default_dtype(dtype_t::f32);

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  int num_channels = parse_with_ss<int>(argv[4]);
  DLINEOUT(addr_zero << " " << std::boolalpha << is_rank_zero << " " << world_size << " " << num_channels);
  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[5]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  int num_threads = parse_with_ss<int>(argv[6]);
  int num_contraction_threads = parse_with_ss<int>(argv[7]);

  std::unique_ptr<server_base_t> server(
    new cpu_tg_server_t(communicator, mem_size, num_threads));

  if(!is_rank_zero) {
    server->listen();
    return 0;
  }

  args_t args(argc-(expected_argc-1), argv+(expected_argc-1));

  main_rank_zero(world_size, server.get(), args);

  server->shutdown();

  return 0;
}

vector<placement_t> make_pls(
  graph_t const& graph,
  int threads)
{
  vector<placement_t> ret;
  for(auto const& part: apart01(graph, threads)) {
    ret.emplace_back(part);
  }

  return ret;
}

int build_input_matrix(
  server_base_t* server,
  uint64_t dn, uint64_t dp, uint64_t dr)
{
  DLINE;
  graph_writer_t writer;
  tensor_t WW = writer.input(vector<uint64_t>{dn,dr});

  tensor_t S = writer.reduction("nr->n", castable_t::add, WW);
  tensor_t W = writer.ew("nr,n->nr", scalarop_t::make_div(), WW, S);

  tensor_t U = writer.input(vector<uint64_t>{dr,dp});
  tensor_t X = writer.matmul(W, U);
  X.save_inplace();

  DLINE;
  {
    dbuffer_t vU = make_dbuffer(dtype_t::f32, dr*dp);
    std::exponential_distribution<float> dist(1);
    std::generate(vU.f32(), vU.f32() + dr*dp, [&dist]{ return dist(random_gen()); });
    server->insert_tensor(U.get_id(), {dr,dp}, vU);
  }

  {
    dbuffer_t vWW = make_dbuffer(dtype_t::f32, dn*dr);
    vWW.random("0.0", "1.0");
    server->insert_tensor(WW.get_id(), {dn,dr}, vWW);
  }

  auto const& graph = writer.get_graph();
  vector<placement_t> pls = make_pls(graph, server->get_cpu_threadpool()->num_runners());
  DLINEOUT("server->execute_graph...");
  server->execute_graph(writer.get_graph(), pls);

  DLINE;
  return X.get_id();
}

struct info_t {
  graph_t graph;
  int X;
  int W_inn;
  int U_inn;
  int W_out;
  int U_out;
  int loss;
};

tensor_t make_loss(
  graph_writer_t& writer,
  tensor_t X,
  tensor_t Xhat,
  tensor_t W,
  tensor_t U,
  tensor_t scale,
  float alpha_w,
  float alpha_u)
{
  tensor_t sqdiff = writer.straight_bew(scalarop_t::make_sub(), X, Xhat);
  sqdiff = writer.ew(scalarop_t::make_square(), sqdiff);
  sqdiff = writer.sum_to_unit(sqdiff);
  uint64_t dn = X.get_shape()()[0];
  sqdiff = writer.ew(scalarop_t::make_scale(scalar_t(float(1/(1.0*dn)))), sqdiff);

  tensor_t wloss = writer.ew(scalarop_t::make_abs(), W);
  wloss = writer.sum_to_unit(wloss);
  wloss = writer.ew(scalarop_t::make_scale(scalar_t(alpha_w)), wloss);

  tensor_t uloss = writer.ew("ij,i->ij", scalarop_t::make_mul(), U, scale);
  uloss = writer.ew(scalarop_t::make_square(), uloss);
  uloss = writer.sum_to_unit(uloss);
  uloss = writer.ew(scalarop_t::make_scale(scalar_t(alpha_u)), uloss);

  tensor_t loss = writer.add(sqdiff, wloss);
  loss = writer.add(loss, uloss);

  return loss;
}

tensor_t update(graph_writer_t& writer, tensor_t V, tensor_t gV, float lr)
{
  scalarop_t grad_update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(default_dtype()),
      scalarop_t::make_scale(scalar_t(lr))
    }
  );

  V = writer.ew("ab,ab->ab", grad_update, V, gV);

  scalarop_t make_positive = scalarop_t::combine(
    scalarop_t::make_max(),
    {
      scalarop_t::make_constant(scalar_t::zero(default_dtype())),
      scalarop_t::make_arg(0)
    });

  V = writer.ew("ab->ab", make_positive, V);

  return V;
}

info_t make_compute(
  uint64_t dn,
  uint64_t dp,
  uint64_t dr,
  float alpha_w,
  float alpha_u,
  float lr)
{
  graph_writer_t writer;

  tensor_t X = writer.input(vector<uint64_t>{dn,dp});
  X.save_inplace();

  tensor_t W_inn = writer.input(vector<uint64_t>{dn,dr});
  tensor_t U_inn = writer.input(vector<uint64_t>{dr,dp});

  tensor_t W = W_inn;
  tensor_t U = U_inn;

  tensor_t Xhat = writer.matmul(W, U);
  tensor_t scale = writer.input(vector<uint64_t>{dr});
  scale.save_inplace();

  tensor_t loss = make_loss(writer, X, Xhat, W, U, scale, alpha_w, alpha_u);

  tensor_t gW = writer.backprop(loss, {W})[0];
  W = update(writer, W, gW, lr);

  Xhat = writer.matmul(W, U);
  loss = make_loss(writer, X, Xhat, W, U, scale, alpha_w, alpha_u);

  tensor_t gU = writer.backprop(loss, {U})[0];
  U = update(writer, U, gU, lr);

  return info_t {
    .graph = writer.get_graph(),
    .X = X.get_id(),
    .W_inn = W_inn.get_id(),
    .U_inn = U_inn.get_id(),
    .W_out = W.get_id(),
    .U_out = U.get_id(),
    .loss = loss.get_id()
  };
}

void main_rank_zero(
  int world_size,
  server_base_t* server,
  args_t& args)
{
  DLINE;
  uint64_t dn = args.get<uint64_t>("dn");
  uint64_t dp = args.get<uint64_t>("dp");
  uint64_t dr = args.get<uint64_t>("dr");

  float alpha_w = args.get<float>("alpha_w") / (1.0 * dn * dr);
  float alpha_u = args.get<float>("alpha_u") / (1.0 * dr * dp);
  float lr      = args.get<float>("learning_rate");

  info_t info = make_compute(dn, dp, dr, alpha_w, alpha_u, lr);

  //////////////////////////////

  DLINE;
  int iX = build_input_matrix(server, dn, dp, dr);
  DLINE;
  int iW = iX + 1;
  int iU = iX + 2;

  // initialize W and U

  DLINE;
  {
    dbuffer_t W = make_dbuffer(dtype_t::f32, dn*dr);
    W.random("0.1", "1.0");
    server->insert_tensor(iW, {dn,dr}, W);
  }

  DLINE;
  {
    dbuffer_t U = make_dbuffer(dtype_t::f32, dr*dp);
    U.random("0.1", "1.0");
    server->insert_tensor(iU, {dr,dp}, U);
  }
}


