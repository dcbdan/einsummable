#include "../src/einsummable/reference.h"
#include "../src/einsummable/memgraph.h"

#include <fstream>

struct random_placement_t {
  placement_t operator()(vector<uint64_t> const& total_shape) {
    vector<partdim_t> partdims;
    for(uint64_t const& n: total_shape) {
      auto const& [beg_,end_] = part_size_rng;
      int p;
      if(end_ > n) {
        p = 1;
      } else {
        p = runif(beg_, end_);
      }
      partdims.push_back(partdim_t::split(n, p));
    }
    partition_t part(partdims);

    return placement_t::random(part, nloc);
  }

  tuple<int,int> part_size_rng;
  int nloc;
};

int main01() {
  uint64_t ni = 3;
  uint64_t nj = 4;
  uint64_t nk = 5;

  dtype_t dtype = dtype_t::f64;
  einsummable_t matmul = einsummable_t::from_matmul(ni,nj,nk, dtype);

  dbuffer_t lhs = make_dbuffer(dtype, ni*nj);
  dbuffer_t rhs = make_dbuffer(dtype, nj*nk);

  lhs.iota(8);
  rhs.iota(100);

  // def f(s0, s1, ni, nj, nk):
  //     x = np.array([i for i in range(s0, s0+ni*nj)]).reshape((ni,nj))
  //     y = np.array([i for i in range(s1, s1+nj*nk)]).reshape((nj,nk))
  //     z = np.dot(x,y)
  //     print(z.reshape(-1))
  // f(0, 100, 3, 3, 3) = [18,321,1242,1254,1266,2169,2190,2211]
  // f(8, 100, 3, 4, 5) = [4110 4148 4186 4224 4262 5830 5884 5938 5992 6046 7550 7620 7690 7760 7830]

  dbuffer_t out = reference_einsummable(matmul, {lhs, rhs});

  std::cout << lhs << " " << rhs << std::endl << out << std::endl;

  return 0;
}

int main02() {
  partition_t partition({
    partdim_t::split(4, 1),
    partdim_t::split(4, 2)
  });

  dtype_t dtype = dtype_t::f64;
  dbuffer_t tensor = make_dbuffer(dtype, 4*4);
  tensor.iota(0);

  tensor_t<dbuffer_t> ptensor = partition_buffer(partition, tensor);
  for(auto const& buffer: ptensor.get()) {
    std::cout << buffer << std::endl;
  }
  return 0;
}

int main03() {
  partition_t partition({
    partdim_t::split(3, 1),
    partdim_t::split(4, 2),
    partdim_t::split(5, 3)
  });

  dtype_t dtype = dtype_t::f16;
  dbuffer_t tensor = make_dbuffer(dtype, 3*4*5);
  tensor.iota(99);

  tensor_t<dbuffer_t> ptensor = partition_buffer(partition, tensor);

  dbuffer_t tensor_again = unpartition_buffer(partition, ptensor);

  if(tensor != tensor_again) {
    std::cout << "Did not correctly undo the partition" << std::endl;
  } else {
    std::cout << "done." << std::endl;
  }

  return 0;
}

int main04() {
  dtype_t dtype = default_dtype();
  dbuffer_t b1 = make_dbuffer(dtype, 10);
  dbuffer_t b2 = b1;
  b2.iota();
  std::cout << b2 << std::endl;
  std::cout << b1 << std::endl;
  return 0;
}

void reblock_test(
  dtype_t dtype,
  placement_t placement_start,
  placement_t placement_finish)
{
  dbuffer_t inn_buffer = make_dbuffer(dtype, 20);
  inn_buffer.random();

  graph_constructor_t graph;
  int gid_inn = graph.insert_input(placement_start, dtype);
  int gid_out = graph.insert_formation(placement_finish, gid_inn);

  dbuffer_t out_buffer = reference_compute_graph(
    graph.graph, { {gid_inn, inn_buffer} })[gid_out];

  std::cout << "-- graph " << std::endl;
  std::cout << "inn " << inn_buffer << std::endl;
  std::cout << "out " << out_buffer << std::endl;
  std::cout << std::endl;

  auto [input_gid_to_tids, output_gid_to_tids, taskgraph] = taskgraph_t::make(
    graph.graph,
    graph.get_placements());

  //taskgraph.print();

  tensor_t<dbuffer_t> inn_pbuffer = partition_buffer(
    placement_start.partition, inn_buffer);
  tensor_t<dbuffer_t> out_pbuffer = partition_buffer(
    placement_finish.partition, out_buffer);

  std::cout << "inn_pbuffer " << inn_pbuffer << std::endl;
  std::cout << "out_pbuffer " << out_pbuffer << std::endl;

  map<int, dbuffer_t> t_input_map = init_buffer_map(
    input_gid_to_tids.at(gid_inn),
    inn_pbuffer);

  auto t_out_map = reference_compute_taskgraph(taskgraph, t_input_map);

  tensor_t<dbuffer_t> t_out_pbuffer = get_partitioned_buffer(
    t_out_map,
    output_gid_to_tids[gid_out]);
  dbuffer_t t_out_buffer = unpartition_buffer(placement_finish.partition, t_out_pbuffer);

  std::cout << "-- taskgraph " << std::endl;
  std::cout << t_out_buffer << std::endl;

  if(t_out_buffer != out_buffer) {
    throw std::runtime_error("reblock test failed");
  }
}

void main05() {
  {
    placement_t placement_start(
      partition_t({ partdim_t::split(20, 5) }),
      tensor_t<int>({5}, {0,0,0,0,0})
    );
    placement_t placement_finish(
      partition_t({ partdim_t::split(20, 3) }),
      tensor_t<int>({3}, {0,0,0})
    );

    reblock_test(dtype_t::c64, placement_start, placement_finish);
  }

  {
    placement_t placement_start(
      partition_t({ partdim_t::split(20, 1) }),
      tensor_t<int>({1}, {0})
    );
    placement_t placement_finish(
      partition_t({ partdim_t::split(20, 2) }),
      tensor_t<int>({2}, {0,0})
    );

    reblock_test(dtype_t::f16, placement_start, placement_finish);
  }

  {
    placement_t placement_start(
      partition_t({ partdim_t::split(20, 2) }),
      tensor_t<int>({2}, {0,0})
    );
    placement_t placement_finish(
      partition_t({ partdim_t::split(20, 1) }),
      tensor_t<int>({1}, {0})
    );

    reblock_test(dtype_t::f32, placement_start, placement_finish);
  }
}

void test_make_taskgraph(
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<int, dbuffer_t> full_inns)
{
  tuple<
    map<int, tensor_t<int> >,
    map<int, tensor_t<int> >,
    taskgraph_t>
    _info = taskgraph_t::make(graph, placements);
  auto const& [inn_to_blocks, out_to_blocks, taskgraph] = _info;

  //taskgraph.print();

  map<int, dbuffer_t> task_inns;
  for(auto [gid, full_buffer]: full_inns) {
    tensor_t<dbuffer_t> pbuffer = partition_buffer(
      placements.at(gid).partition,
      full_buffer);
    fill_buffer_map(task_inns, inn_to_blocks.at(gid), pbuffer);
  }

  map<int, dbuffer_t> full_outs = reference_compute_graph(graph, full_inns);
  map<int, dbuffer_t> task_outs = reference_compute_taskgraph(taskgraph, task_inns);

  for(auto const& [gid, full_buffer_via_graph]: full_outs) {
    tensor_t<dbuffer_t> t_part_buffer =
      get_partitioned_buffer(task_outs, out_to_blocks.at(gid));
    tensor_t<dbuffer_t> part_buffer =
      partition_buffer(placements.at(gid).partition, full_buffer_via_graph);

    auto const& tids  = out_to_blocks.at(gid).get();
    auto const& t_vec = t_part_buffer.get();
    auto const& vec   = part_buffer.get();
    for(int i = 0; i != vec.size(); ++i) {
      auto const& tid = tids[i];
      auto const& t   = t_vec[i];
      auto const& u   = vec[i];
      if(!is_close(t, u)) {
        std::cout << tid << std::endl << t << std::endl << u << std::endl;
        throw std::runtime_error("make_taskgraph_test fail");
      }
    }
    // Alternatively:
    //   buffer_t full_buffer_via_taskgraph =
    //     unpartition_buffer(graph.placements.at(gid).partition, part_buffer);
    //   if(!is_close(full_buffer_via_graph, full_buffer_via_taskgraph)) {
    //     throw std::runtime_error("make_taskgraph_test fail");
    //   }
  }
}

void test_make_taskgraph(
  graph_constructor_t const& graph,
  map<int, dbuffer_t> full_inns)
{
  return test_make_taskgraph(graph.graph, graph.get_placements(), full_inns);
}

void test_make_memgraph_without_evict(
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<int, dbuffer_t> full_inns)
{
  graph.print();
  for(int i = 0; i != graph.nodes.size(); ++i) {
    DOUT(i << ": " << placements[i].partition);
  }

  tuple<
    map<int, tensor_t<int> >,
    map<int, tensor_t<int> >,
    taskgraph_t>
    _info0 = taskgraph_t::make(graph, placements);
  auto const& [inn_to_blocks, out_to_blocks, taskgraph] = _info0;

  {
    std::cout << "Printing to exp_reference_taskgraph.gv" << std::endl;
    std::ofstream f("exp_reference_taskgraph.gv");
    taskgraph.print_graphviz(f);
  }

  int num_locs = taskgraph.num_locs();

  // have everyone share the same cache
  vector<int> compute_loc_to_cache(num_locs, 0);

  tuple<
    map<int, mem_t>, // input -> mem
    map<int, mem_t>, // save -> mem
    memgraph_t>
    _info1 = memgraph_t::make_without_evict(taskgraph, compute_loc_to_cache);
  auto const& [task_inn_to_mem, task_out_to_mem, memgraph] = _info1;

  // allocate a blob of memory at each compute location
  vector<buffer_t> loc_buffers;
  vector<uint64_t> mem_sizes = memgraph.mem_sizes();
  for(uint64_t const& mem_sz: mem_sizes) {
    loc_buffers.push_back(std::make_shared<buffer_holder_t>(mem_sz));
  }

  // Initialize the taskgraph inputs and then
  // copy into the location wise buffers
  {
    map<int, dbuffer_t> task_inns;
    for(auto [gid, full_buffer]: full_inns) {
      tensor_t<dbuffer_t> pbuffer = partition_buffer(
        placements.at(gid).partition,
        full_buffer);
      fill_buffer_map(task_inns, inn_to_blocks.at(gid), pbuffer);
    }

    for(auto const& [tid, dbuffer]: task_inns) {
      auto buffer = dbuffer.data;
      auto const& [offset, size] = task_inn_to_mem.at(tid);
      if(size != buffer->size) {
        throw std::runtime_error("maybe invalid task_inn_to_mem");
      }
      int loc = taskgraph.nodes[tid].op.out_loc();
      std::copy(
        buffer->data, buffer->data + size,
        loc_buffers[loc]->data + offset);
    }
  }

  // compute the reference implementation
  map<int, dbuffer_t> full_outs = reference_compute_graph(graph, full_inns);

  //{
  //  std::cout << "Printing to exp_reference_memgraph.gv" << std::endl;
  //  std::ofstream f("exp_reference_memgraph.gv");
  //  memgraph.print_graphviz(f);
  //}

  reference_compute_memgraph(memgraph, loc_buffers);

  for(auto const& [gid, full_buffer]: full_outs) {
    tensor_t<dbuffer_t> part_buffer =
      partition_buffer(placements.at(gid).partition, full_buffer);

    auto const& tids = out_to_blocks.at(gid).get();
    auto const& vec  = part_buffer.get();
    for(int i = 0; i != vec.size(); ++i) {
      int       const& tid = tids[i];
      dbuffer_t const& from_graph = vec[i];

      // where in the loc_buffers the result is
      int loc = taskgraph.nodes[tid].op.out_loc();
      auto const& [offset, size] = task_out_to_mem.at(tid);

      dbuffer_t from_memgraph = dbuffer_t(
        from_graph.dtype,
        make_buffer_reference(loc_buffers[loc]->data + offset, size));

      if(!is_close(from_graph, from_memgraph)) {
        std::cout << "expected: " << from_graph    << std::endl;
        std::cout << "actual:   " << from_memgraph << std::endl;
        throw std::runtime_error("make memgraph without evict test fail");
      }
    }
  }
}

void test_make_memgraph_without_evict(
  graph_constructor_t const& graph,
  map<int, dbuffer_t> full_inns)
{
  return test_make_memgraph_without_evict(
    graph.graph, graph.get_placements(), full_inns);
}

// Here, obvious matmul means
// 1. block the i,j,k dimensions,
// 2. join i,j,k and do matmul at each block
// 3. agg out j to i,k
void test_obvious_matmul(int pi, int pj, int pk) {
  graph_constructor_t graph;

  uint64_t ni = 10;
  uint64_t nj = 10;
  uint64_t nk = 10;

  partdim_t pdi = partdim_t::split(ni, pi);
  partdim_t pdj = partdim_t::split(nj, pj);
  partdim_t pdk = partdim_t::split(nk, pk);

  int id_lhs = graph.insert_input(partition_t({pdi,pdj}));
  int id_rhs = graph.insert_input(partition_t({pdj,pdk}));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    partition_t({pdi, pdk, pdj}),
    matmul,
    {id_lhs, id_rhs});

  int id_save = graph.insert_formation(
    partition_t({pdi, pdk}),
    id_join,
    true);

  dbuffer_t buffer_lhs = make_dbuffer(default_dtype(), ni*nj);
  buffer_lhs.iota(-10);

  dbuffer_t buffer_rhs = make_dbuffer(default_dtype(), nj*nk);
  buffer_rhs.iota(-20);

  map<int, dbuffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };

  test_make_memgraph_without_evict(graph, inns);
}

void test_obvious_same_input_matmul(int pi, int pj, int pk) {
  graph_constructor_t graph;

  uint64_t ni = 10;
  uint64_t nj = ni;
  uint64_t nk = nj;

  partdim_t pdi = partdim_t::split(ni, pi);
  partdim_t pdj = partdim_t::split(nj, pj);
  partdim_t pdk = partdim_t::split(nk, pk);

  int id_inn = graph.insert_input(partition_t({pdi,pdj}));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    partition_t({pdi, pdk, pdj}),
    matmul,
    {id_inn, id_inn});

  int id_save = graph.insert_formation(
    partition_t({pdi, pdk}),
    id_join,
    true);

  dbuffer_t buffer_inn = make_dbuffer(default_dtype(), ni*nj);
  buffer_inn.iota(-10);

  map<int, dbuffer_t> inns{ {id_inn, buffer_inn} };
  test_make_taskgraph(graph, inns);
}

void test_obvious_random_loc_matmul(int pi, int pj, int pk, int nloc) {
  graph_constructor_t graph;

  uint64_t ni = 10;
  uint64_t nj = 10;
  uint64_t nk = 10;

  partdim_t pdi = partdim_t::split(ni, pi);
  partdim_t pdj = partdim_t::split(nj, pj);
  partdim_t pdk = partdim_t::split(nk, pk);

  int id_lhs = graph.insert_input(placement_t::random(partition_t({pdi,pdj}), nloc));
  int id_rhs = graph.insert_input(placement_t::random(partition_t({pdj,pdk}), nloc));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    placement_t::random(partition_t({pdi, pdk, pdj}), nloc),
    matmul,
    {id_lhs, id_rhs});

  int id_save = graph.insert_formation(
    placement_t::random(partition_t({pdi, pdk}), nloc),
    id_join,
    true);

  dbuffer_t buffer_lhs = make_dbuffer(default_dtype(), ni*nj);
  buffer_lhs.iota(1); //-10);

  dbuffer_t buffer_rhs = make_dbuffer(default_dtype(), nj*nk);
  buffer_rhs.iota(7); // -20);

  map<int, dbuffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };
  test_make_memgraph_without_evict(graph, inns);
}

void test_random_matmul() {
  int nloc = 20;
  random_placement_t random_placement { {1, 10}, nloc };

  graph_constructor_t graph;

  uint64_t ni = 10;
  uint64_t nj = 10;
  uint64_t nk = 10;

  int id_lhs = graph.insert_input(random_placement({ni,nj}));
  int id_rhs = graph.insert_input(random_placement({nj,nk}));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    random_placement({ni,nk,nj}),
    matmul,
    {id_lhs, id_rhs});

  int id_save = graph.insert_formation(
    random_placement({ni,nk}),
    id_join,
    true);

  dbuffer_t buffer_lhs = make_dbuffer(default_dtype(), ni*nj);
  buffer_lhs.iota(-10);

  dbuffer_t buffer_rhs = make_dbuffer(default_dtype(), nj*nk);
  buffer_rhs.iota(-20);

  map<int, dbuffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };

  test_make_taskgraph(graph, inns);

  test_make_memgraph_without_evict(graph, inns);
}

void test_random_matmul_then_unary_ew(scalarop_t unary_scalar_op) {
  if(!unary_scalar_op.is_unary()) {
    throw std::runtime_error("expecting a unary op");
  }
  dtype_t inn_dtype = unary_scalar_op.inn_dtype(0).value();
  dtype_t out_dtype = unary_scalar_op.out_dtype();

  int nloc = 3;
  random_placement_t random_placement { {1, 10}, nloc };

  graph_constructor_t graph;

  uint64_t ni = 10;
  uint64_t nj = 10;
  uint64_t nk = 10;

  int id_lhs = graph.insert_input(random_placement({ni,nj}));
  int id_rhs = graph.insert_input(random_placement({nj,nk}));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk, inn_dtype);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    random_placement({ni,nk,nj}),
    matmul,
    {id_lhs, id_rhs});

  einsummable_t unary_es(
    {ni,nk},
    { {0,1} },
    2,
    unary_scalar_op);

  auto const& pds = graph.placements.at(id_join).partition;
  partition_t part_unary(vector<partdim_t>(
    pds.partdims.begin(),
    pds.partdims.begin() + 2));

  int id_unary = graph.insert_einsummable(
    placement_t::random(part_unary, nloc),
    unary_es,
    {id_join});

  int id_save = graph.insert_formation(
    random_placement({ni,nk}),
    id_unary,
    true);

  dbuffer_t buffer_lhs = make_dbuffer(inn_dtype, ni*nj);
  buffer_lhs.iota(-10);

  dbuffer_t buffer_rhs = make_dbuffer(inn_dtype, nj*nk);
  buffer_rhs.iota(-20);

  map<int, dbuffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };

  test_make_memgraph_without_evict(graph, inns);
}

void main06(int argc, char** argv) {
  if(argc != 4) {
    throw std::runtime_error("usage: pi pj pk");
  }
  int pi = parse_with_ss<int>(argv[1]);
  int pj = parse_with_ss<int>(argv[2]);
  int pk = parse_with_ss<int>(argv[3]);
  test_obvious_matmul(pi, pj, pk);
}

void main07(int argc, char** argv) {
  if(argc != 4) {
    throw std::runtime_error("usage: pi pj pk");
  }
  int pi = parse_with_ss<int>(argv[1]);
  int pj = parse_with_ss<int>(argv[2]);
  int pk = parse_with_ss<int>(argv[3]);
  test_obvious_same_input_matmul(pi, pj, pk);
}

void main08(int argc, char** argv) {
  if(argc != 6) {
    throw std::runtime_error("usage: pi pj pk loc seed");
  }
  int pi   = parse_with_ss<int>(argv[1]);
  int pj   = parse_with_ss<int>(argv[2]);
  int pk   = parse_with_ss<int>(argv[3]);
  int nloc = parse_with_ss<int>(argv[4]);
  int seed = parse_with_ss<int>(argv[5]);

  set_seed(seed);

  test_obvious_random_loc_matmul(pi, pj, pk, nloc);
}

void main09(int argc, char** argv) {
  if(argc == 1) {
    // pass
  } else if(argc == 2) {
    int seed = parse_with_ss<int>(argv[1]);
    set_seed(seed);
  } else {
    throw std::runtime_error("usage: [optional]seed");
  }

  test_random_matmul();
}

void main10() {
  for(int seed = 0; seed != 1000; ++seed) {
    std::cout << "seed: " << seed << std::endl;
    set_seed(seed);
    test_random_matmul();
  }
}

void test_3d_matmul(int pi, int pj, int pk, int nloc)
{
  uint64_t di = 10;
  uint64_t dj = 10;
  uint64_t dk = 10;

  uint64_t ni = pi*di;
  uint64_t nj = pj*dj;
  uint64_t nk = pk*dk;

  graph_constructor_t graph = three_dimensional_matrix_multiplication(pi,pj,pk,di,dj,dk,nloc);

  dbuffer_t buffer0 = make_dbuffer(default_dtype(), ni*nj);
  buffer0.iota(-10);

  dbuffer_t buffer1 = make_dbuffer(default_dtype(), nj*nk);
  buffer1.iota(-20);

  vector<int> inputs = graph.graph.get_inputs();
  map<int, dbuffer_t> inns{ {inputs[0], buffer0}, {inputs[1], buffer1} };

  test_make_memgraph_without_evict(graph, inns);
}

void test_matmul_reference(uint64_t di, uint64_t dj, uint64_t dk) {
  dtype_t dtype = dtype_t::f32;
  dbuffer_t lhs = make_dbuffer(dtype, di*dj);
  lhs.random();

  dbuffer_t rhs = make_dbuffer(dtype, dj*dk);
  rhs.random();

  dbuffer_t out_true = make_dbuffer(dtype, di*dk);
  out_true.zeros();

  float* ll = lhs.f32();
  float* rr = rhs.f32();
  float* oo = out_true.f32();

  for(int i = 0; i != di; ++i) {
  for(int j = 0; j != dj; ++j) {
  for(int k = 0; k != dk; ++k) {
    oo[i*dk + k] += ll[i*dj + j] * rr[j*dk + k];
  }}}

  einsummable_t matmul = einsummable_t::from_matmul(di, dj, dk, dtype);
  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  std::cout << out_true << std::endl;
  std::cout << out_ref  << std::endl;
}

void main11(int argc, char** argv) {
  if(argc != 5) {
    throw std::runtime_error("usage: pi pj pk nproc");
  }
  int pi = parse_with_ss<int>(argv[1]);
  int pj = parse_with_ss<int>(argv[2]);
  int pk = parse_with_ss<int>(argv[3]);
  int np = parse_with_ss<int>(argv[4]);
  test_3d_matmul(pi, pj, pk, np);
}

void main12() {
  // Test the reference_concat works allright

  auto run = [](int dim, vector<uint64_t> shape_template) {
    dtype_t dtype = dtype_t::f32;
    shape_template[dim] = 1;
    uint64_t n = product(shape_template);

    vector<uint64_t> ds { 3, 4, 5, 6 };
    vector<dbuffer_t> bs;
    for(int i = 0; i != ds.size(); ++i) {
      auto const& d = ds[i];
      bs.push_back(make_dbuffer(dtype, n*d));
      bs.back().fill(scalar_t(float(1.0*i)));
    }

    vector<vector<uint64_t>> shapes;
    for(auto const& d: ds) {
      shapes.push_back(shape_template);
      shapes.back()[dim] = d;
    }

    concat_t concat(dim, dtype, shapes);

    dbuffer_t out = reference_concat(concat, bs);

    DOUT(out);
  };

  run(0, {1});
  run(0, {1,2});
  run(1, {2,1});
  run(1, {2,1,2});
}

void test_random_concat(
  int dim,
  vector<uint64_t> shape_template,
  int n_inn)
{
  dtype_t dtype;
  int dd = runif(4);
  if(dd == 0) {
    dtype = dtype_t::f16;
  } else if(dd == 1) {
    dtype = dtype_t::f32;
  } else if(dd == 2) {
    dtype = dtype_t::f64;
  } else if(dd == 3) {
    dtype = dtype_t::c64;
  }

  shape_template[dim] = 1;
  uint64_t n = product(shape_template);

  graph_writer_t w;

  using id_t = graph_writer_t::tensor_t;

  map<int, dbuffer_t> inn_tensors;
  vector<id_t> inns;
  for(int i = 0; i != n_inn; ++i) {
    auto shape = shape_template;
    shape[dim] = runif(10,20);
    inns.push_back(w.input(shape, dtype));

    inn_tensors.insert({
      inns.back().get_id(),
      make_dbuffer(dtype, product(shape))});
  }

  for(auto& [_, buffer]: inn_tensors) {
    buffer.random();
  }

  id_t x = w.concat(dim, inns);

  x.save();

  int nloc = 3;
  random_placement_t random_placement { {1, 10}, nloc };

  graph_t g = w.get_graph();
  vector<placement_t> pls;
  for(int gid = 0; gid != g.nodes.size(); ++gid) {
    pls.push_back(random_placement(g.nodes[gid].op.shape()));
  }

  test_make_taskgraph(g, pls, inn_tensors);
}

void main13() {
  graph_writer_t w;

  using id_t = graph_writer_t::tensor_t;

  dtype_t dtype = dtype_t::f32;

  id_t a = w.input({4,3}, dtype);
  id_t b = w.input({4,5}, dtype);
  id_t c = w.input({5,3}, dtype);

  map<int, dbuffer_t> inns;
  inns.insert({a.get_id(), make_dbuffer(dtype, 4*3)});
  inns.insert({b.get_id(), make_dbuffer(dtype, 4*5)});
  inns.insert({c.get_id(), make_dbuffer(dtype, 5*3)});

  for(auto& [_, buffer]: inns) {
    buffer.random();
  }

  id_t x = w.concat(1, {a,b});
  id_t y = w.concat(0, {a,c});

  x.save();
  y.save();

  graph_t g = w.get_graph();

  DOUT("singleton concats...");
  {
    // singleton placements
    auto pls = g.make_singleton_placement();
    test_make_taskgraph(g, pls, inns);
  }

  DOUT("concats with partitions along concat...");
  {
    // singleton input placements, but preserve partition on
    // the concats of a
    auto pls = g.make_singleton_placement();

    for(int id = 0; id != g.nodes.size(); ++id) {
      auto const& node = g.nodes[id];
      if(node.op.is_concat()) {
        pls[id] = concat_split_placement(pls[id], node.op.get_concat());
      }
    }

    test_make_taskgraph(g, pls, inns);
  }
  DOUT("all done.");
}

void main14() {
  for(int i = 0; i != 100; ++i) {
    DOUT(i);
    set_seed(i);
    int n_inn = runif(2,5);
    int dim = runif(4);
    test_random_concat(dim, {20,19,18,17}, n_inn);
  }
}

void test_random_goofy_ff() {
  graph_writer_t writer;
  using id_t = graph_writer_t::tensor_t;

  uint64_t bsz = 3;
  uint64_t d0 = 4;
  uint64_t d1 = 5;
  uint64_t d2 = 6;
  uint64_t d3 = 7;

  uint64_t d01 = d0*d1;
  uint64_t d12 = d1*d2;
  uint64_t d22 = d2*d2;
  uint64_t d32 = d3*d2;
  uint64_t d33 = d3*d3;

  dtype_t dtype = dtype_t::f32;

  id_t x = writer.input({bsz,d0,d1}, dtype).view({bsz,d01});
  id_t w0 = writer.input({d0,d1,d3,d2}, dtype).view({d01,d32});
  id_t w1 = writer.input({d3,d2}, dtype);

  map<int,dbuffer_t> inns;
  for(id_t id: vector<id_t>{x,w0,w1}) {
    int gid = id.get_id();
    dbuffer_t buffer = make_dbuffer(dtype, product(id.get_shape()));
    buffer.random();
    inns.insert({gid, buffer});
  }

  id_t y = writer.matmul(x, w0).view({bsz, d3, d2});
  id_t z = writer.matmul(y, w1.transpose(0,1)); // bsz,d3,d3
  y = writer.concat(2, {y, z}); // bsz,d3,d33
  y = writer.reduction("bxy->bx", castable_t::add, y); // bsz,d1

  y.save();

  int nloc = 3;
  random_placement_t random_placement { {1, 10}, nloc };

  graph_t g = writer.get_graph();
  vector<placement_t> pls;
  for(int gid = 0; gid != g.nodes.size(); ++gid) {
    pls.push_back(random_placement(g.nodes[gid].op.shape()));
  }

  test_make_taskgraph(g, pls, inns);
}

int main(int argc, char** argv) {
  //main09(argc, argv);
  //main10();
  //main11(argc, argv);
  //set_seed(0);
  //test_obvious_random_loc_matmul(5,5,5,5);

  set_seed(0);
  test_random_matmul_then_unary_ew(scalarop_t::make_increment(scalar_t(float(0.77))));

  //main13();
  //main14();

  //set_seed(0);
  //test_random_concat(0, {20,19,18}, 3);

  //for(int i = 0; i != 1000; ++i) {
  //  DOUT(i);
  //  set_seed(i);
  //  test_random_goofy_ff();
  //}

  //set_seed(1);
  //test_random_goofy_ff();

  //main02();
  //main03();
  //main04();
  //main05();
}
