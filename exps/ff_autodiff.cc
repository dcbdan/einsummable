#include "../src/base/setup.h"
#include "../src/base/args.h"

#include "../src/einsummable/gwriter.h"
#include "../src/server/cpu/server.h"

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

// loss_id:      What to take the backprop against. Need not be saved
// inspect_ids:  tensors that get computed and should be saved so the user
//               can inspect them
// data_ids:     tensors that get inserted by the user before each iteration
//               (e.g. data matrix x and correct results y)
// constant_ids: input tensors that never get changed and must be insert by
//               the user before the first iteration
// weight_ids:   the tensors that get updated via update(weight, grad)
struct trainer_t {
  trainer_t(
    server_base_t* server,
    graph_t const& init_graph,
    int loss_id,
    vector<int> const& inspect_ids,
    vector<int> const& data_ids,
    vector<int> const& constant_ids,
    vector<int> const& weight_ids,
    scalarop_t update /* weight elem , grad elem -> new eight elem */)
    : server(server)
  {
    graph_t graph = init_graph;
    map<int, string> colors;
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
      graph.nodes[gid].op.set_save(false);
      colors.insert({gid, "yellow"});
    }

    for(int const& inspect_id: inspect_ids) {
      graph.nodes[inspect_id].op.set_save(true);
    }
    for(int const& constant_id: constant_ids) {
      graph.nodes[constant_id].op.set_save(true);
    }

    vector<int> grad_ids = graph.backprop(loss_id, weight_ids);

    vector<int> updated_weights;
    updated_weights.reserve(grad_ids.size());
    for(auto [weight, grad]: vector_zip(weight_ids, grad_ids)) {
      int updated_weight = graph.insert_einsummable(
        make_einsummable_update(update, graph.out_shape(weight)),
        {weight, grad});
      graph.nodes[updated_weight].op.set_save(true);
      updated_weights.push_back(updated_weight);
    }

    {
      std::ofstream out("g.gv");
      graph.print_graphviz(out, colors);
      DOUT("printed g.gv");
    }

    // TODO: here we should come up with good placements, somehow
    vector<placement_t> placements;
    placements.reserve(graph.nodes.size());
    for(auto const& node: graph.nodes) {
      placements.emplace_back(partition_t::singleton(node.op.shape()));
    }

    auto [inn_tids, out_tids, taskgraph_] = taskgraph_t::make(graph, placements);
    taskgraph = std::move(taskgraph_);

    // Make sure that inn_ids == weights ++ data_ids ++ constant_ids
    {
      set<int> inn_gids;
      for(auto const& [inn_gid, _]: inn_tids) {
        inn_gids.insert(inn_gid);
      }
      set<int> inn_gids_(weight_ids.begin(), weight_ids.end());
      set_union_inplace(inn_gids_, set<int>(data_ids.begin(), data_ids.end()));
      set_union_inplace(inn_gids_, set<int>(constant_ids.begin(), constant_ids.end()));
      if(inn_gids != inn_gids_) {
        throw std::runtime_error("invalid input gid set in trainer initialization");
      }
    }

    for(auto const& [inn_gid, tids]: inn_tids) {
      inn_remap.insert({
        inn_gid,
        relation_t {
          .dtype     = graph.out_dtype(inn_gid),
          .placement = placements.at(inn_gid),
          .tids      = tids
        }
      });
    }

    for(auto const& [weight, updated_weight]: vector_zip(weight_ids, updated_weights)) {
      after_execution_map.insert({
        updated_weight,
        relation_t {
          .dtype     = graph.out_dtype(updated_weight),
          .placement = placements.at(weight),
          .tids      = out_tids.at(updated_weight)
        }
      });

      out_remap_rels.insert({
        updated_weight,
        relation_t {
          .dtype     = graph.out_dtype(weight),
          .placement = placements.at(weight),
          .tids      = inn_tids.at(weight)
        }
      });
      out_remap_gids.emplace_back(updated_weight, weight);
    }

    // remap info for constant ids and inspect ids are just to make sure things
    // are not deleted
    for(int const& id: vector_concatenate(constant_ids, inspect_ids)) {
      relation_t rel {
        .dtype     = graph.out_dtype(id),
        .placement = placements.at(id),
        .tids      = out_tids.at(id)
      };

      after_execution_map.insert({id, rel});

      out_remap_rels.insert({id, rel});

      out_remap_gids.emplace_back(id, id);
    }
  }

  void operator()() {
    // prepare: Verify that data ids, constant ids and weight ids are in the server.
    //          and do a remap to give them the correct placement.
    server->remap(inn_remap);
    // Note: this will delete any inspect tensors from the previous iteration

    // execute: Run the graph
    server->execute(taskgraph, after_execution_map);

    // remap: make sure that
    //        1) the updated weights are at the init weights
    //        2) constant tensors and inspect tensors are not deleted
    server->remap(out_remap_rels);
    server->remap_gids(out_remap_gids);
  }

  relation_t const& get_input_relation(int id) {
    return inn_remap.at(id);
  }

private:
  static einsummable_t make_einsummable_update(
    scalarop_t update,
    vector<uint64_t> const& shape)
  {
    int rank = shape.size();
    return einsummable_t(
      shape,
      { vector_iota<int>(rank), vector_iota<int>(rank) },
      rank,
      update);
  }

private:
  server_base_t* server;

  taskgraph_t taskgraph;

  map<int, relation_t> inn_remap;

  map<int, relation_t> after_execution_map;

  map<int, relation_t> out_remap_rels;
  vector<tuple<int, int>> out_remap_gids;
};

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

  communicator_t communicator(addr_zero, is_rank_zero, world_size);
  cpu_mg_server_t server(communicator, mem_size, num_threads);

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

  trainer_t trainer(
    &server,
    graph,
    info.loss_id,           // loss
    {info.loss_id},         // inspect
    {info.x_id, info.y_id}, // data
    {},                     // constants
    info.w_ids,             // weights
    grad_update);
  // Here, we'll vary x at every iteration but
  // y will be held constant

  /////////////////
  // > initialize the weights
  for(int const& gid: info.w_ids) {
    auto shape = graph.out_shape(gid);
    dbuffer_t data = make_dbuffer(dtype_t::f32, product(shape));
    data.random("-0.00001", "0.00001");
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
      DOUT("avg loss at iter " << iter << ": " << (loss_so_far / 100.0));
      loss_so_far = 0.0;
    }
  }
}


