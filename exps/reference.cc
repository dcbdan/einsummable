#include "../src/einsummable/reference.h"
#include "../src/einsummable/memgraph.h"

#include <fstream>

int main01() {
  uint64_t ni = 3;
  uint64_t nj = 4;
  uint64_t nk = 5;

  einsummable_t matmul = einsummable_t::from_matmul(ni,nj,nk);

  buffer_t lhs = std::make_shared<buffer_holder_t>(ni*nj);
  buffer_t rhs = std::make_shared<buffer_holder_t>(nj*nk);

  lhs->iota(8);
  rhs->iota(100);

  // def f(s0, s1, ni, nj, nk):
  //     x = np.array([i for i in range(s0, s0+ni*nj)]).reshape((ni,nj))
  //     y = np.array([i for i in range(s1, s1+nj*nk)]).reshape((nj,nk))
  //     z = np.dot(x,y)
  //     print(z.reshape(-1))
  // f(0, 100, 3, 3, 3) = [18,321,1242,1254,1266,2169,2190,2211]
  // f(8, 100, 3, 4, 5) = [4110 4148 4186 4224 4262 5830 5884 5938 5992 6046 7550 7620 7690 7760 7830]

  buffer_t out = reference_einsummable(matmul, {lhs, rhs});

  std::cout << lhs << " " << rhs << " " << out << std::endl;

  return 0;
}

int main02() {
  partition_t partition({
    partdim_t::split(4, 1),
    partdim_t::split(4, 2)
  });

  buffer_t tensor = std::make_shared<buffer_holder_t>(4*4);
  tensor->iota(0);

  tensor_t<buffer_t> ptensor = partition_buffer(partition, tensor);
  for(auto const& buffer_t: ptensor.get()) {
    std::cout << buffer_t << std::endl;
  }
  return 0;
}

int main03() {
  partition_t partition({
    partdim_t::split(3, 1),
    partdim_t::split(4, 2),
    partdim_t::split(5, 3)
  });

  buffer_t tensor = std::make_shared<buffer_holder_t>(3*4*5);
  tensor->iota(99);

  tensor_t<buffer_t> ptensor = partition_buffer(partition, tensor);

  buffer_t tensor_again = unpartition_buffer(partition, ptensor);

  if(!vector_equal(tensor->as_vector(), tensor_again->as_vector())) {
    std::cout << "Did not correctly undo the partition" << std::endl;
  } else {
    std::cout << "done." << std::endl;
  }

  return 0;
}

int main04() {
  buffer_t b1 = std::make_shared<buffer_holder_t>(10);
  buffer_t b2 = b1;
  b2->iota();
  std::cout << b2 << std::endl;
  std::cout << b1 << std::endl;
  return 0;
}

void reblock_test(
  placement_t placement_start,
  placement_t placement_finish)
{
  buffer_t inn_buffer = std::make_shared<buffer_holder_t>(20);
  inn_buffer->iota(0);

  graph_t graph;
  int gid_inn = graph.insert_input(placement_start);
  int gid_out = graph.insert_formation(placement_finish, gid_inn);

  buffer_t out_buffer = reference_compute_graph(graph, { {gid_inn, inn_buffer} })[gid_out];

  std::cout << "-- graph " << std::endl;
  std::cout << "inn " << inn_buffer << std::endl;
  std::cout << "out " << out_buffer << std::endl;
  std::cout << std::endl;

  auto [input_gid_to_tids, output_gid_to_tids, taskgraph] = taskgraph_t::make(graph);

  taskgraph.print();

  tensor_t<buffer_t> inn_pbuffer = partition_buffer(
    placement_start.partition, inn_buffer);
  tensor_t<buffer_t> out_pbuffer = partition_buffer(
    placement_finish.partition, out_buffer);

  std::cout << "inn_pbuffer " << inn_pbuffer << std::endl;
  std::cout << "out_pbuffer " << out_pbuffer << std::endl;

  map<int, buffer_t> t_input_map = init_buffer_map(
    input_gid_to_tids.at(gid_inn),
    inn_pbuffer);

  auto t_out_map = reference_compute_taskgraph(taskgraph, t_input_map);

  tensor_t<buffer_t> t_out_pbuffer = get_partitioned_buffer(
    t_out_map,
    output_gid_to_tids[gid_out]);
  buffer_t t_out_buffer = unpartition_buffer(placement_finish.partition, t_out_pbuffer);

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

    reblock_test(placement_start, placement_finish);
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

    reblock_test(placement_start, placement_finish);
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

    reblock_test(placement_start, placement_finish);
  }
}

void test_make_taskgraph(
  graph_t const& graph,
  map<int, buffer_t> full_inns)
{
  tuple<
    map<int, tensor_t<int> >,
    map<int, tensor_t<int> >,
    taskgraph_t>
    _info = taskgraph_t::make(graph);
  auto const& [inn_to_blocks, out_to_blocks, taskgraph] = _info;

  taskgraph.print();

  map<int, buffer_t> task_inns;
  for(auto [gid, full_buffer]: full_inns) {
    tensor_t<buffer_t> pbuffer = partition_buffer(
      graph.nodes[gid].placement.partition,
      full_buffer);
    fill_buffer_map(task_inns, inn_to_blocks.at(gid), pbuffer);
  }

  map<int, buffer_t> full_outs = reference_compute_graph(graph, full_inns);
  map<int, buffer_t> task_outs = reference_compute_taskgraph(taskgraph, task_inns);

  for(auto const& [gid, full_buffer_via_graph]: full_outs) {
    tensor_t<buffer_t> t_part_buffer =
      get_partitioned_buffer(task_outs, out_to_blocks.at(gid));
    tensor_t<buffer_t> part_buffer =
      partition_buffer(graph.nodes[gid].placement.partition, full_buffer_via_graph);

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
    //     unpartition_buffer(graph.nodes[gid].placement.partition, part_buffer);
    //   if(!is_close(full_buffer_via_graph, full_buffer_via_taskgraph)) {
    //     throw std::runtime_error("make_taskgraph_test fail");
    //   }
  }
}

void test_make_memgraph_without_evict(
  graph_t const& graph,
  map<int, buffer_t> full_inns)
{
  tuple<
    map<int, tensor_t<int> >,
    map<int, tensor_t<int> >,
    taskgraph_t>
    _info0 = taskgraph_t::make(graph);
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
    map<int, buffer_t> task_inns;
    for(auto [gid, full_buffer]: full_inns) {
      tensor_t<buffer_t> pbuffer = partition_buffer(
        graph.nodes[gid].placement.partition,
        full_buffer);
      fill_buffer_map(task_inns, inn_to_blocks.at(gid), pbuffer);
    }

    for(auto const& [tid, buffer]: task_inns) {
      auto const& [offset, size] = task_inn_to_mem.at(tid);
      if(size != buffer->size) {
        throw std::runtime_error("maybe invalid task_inn_to_mem");
      }
      int loc = taskgraph.nodes[tid].op.output_loc();
      std::copy(
        buffer->data, buffer->data + size,
        loc_buffers[loc]->data + offset);
    }
  }

  // compute the reference implementation
  map<int, buffer_t> full_outs = reference_compute_graph(graph, full_inns);

  {
    std::cout << "Printing to exp_reference_memgraph.gv" << std::endl;
    std::ofstream f("exp_reference_memgraph.gv");
    memgraph.print_graphviz(f);
    memgraph.print_graphviz(std::cout);
  }

  reference_compute_memgraph(memgraph, loc_buffers);

  for(auto const& [gid, full_buffer]: full_outs) {
    tensor_t<buffer_t> part_buffer =
      partition_buffer(graph.nodes[gid].placement.partition, full_buffer);

    auto const& tids = out_to_blocks.at(gid).get();
    auto const& vec  = part_buffer.get();
    for(int i = 0; i != vec.size(); ++i) {
      int      const& tid = tids[i];
      buffer_t const& from_graph = vec[i];

      // where in the loc_buffers the result is
      int loc = taskgraph.nodes[tid].op.output_loc();
      auto const& [offset, size] = task_out_to_mem.at(tid);

      buffer_t from_memgraph =
        make_buffer_reference(loc_buffers[loc]->data + offset, size);

      if(!is_close(from_graph, from_memgraph)) {
        std::cout << "expected: " << from_graph    << std::endl;
        std::cout << "actual:   " << from_memgraph << std::endl;
        throw std::runtime_error("make memgraph without evict test fail");
      }
    }
  }

}

// Here, obvious matmul means
// 1. block the i,j,k dimensions,
// 2. join i,j,k and do matmul at each block
// 3. agg out j to i,k
void test_obvious_matmul(int pi, int pj, int pk) {
  graph_t graph;

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

  graph.print();

  buffer_t buffer_lhs = std::make_shared<buffer_holder_t>(ni*nj);
  buffer_lhs->iota(-10);

  buffer_t buffer_rhs = std::make_shared<buffer_holder_t>(nj*nk);
  buffer_rhs->iota(-20);

  map<int, buffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };

  // TODO uncomment
  //test_make_taskgraph(graph, inns);

  test_make_memgraph_without_evict(graph, inns);
}

void test_obvious_same_input_matmul(int pi, int pj, int pk) {
  graph_t graph;

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

  graph.print();

  buffer_t buffer_inn = std::make_shared<buffer_holder_t>(ni*nj);
  buffer_inn->iota(-10);

  map<int, buffer_t> inns{ {id_inn, buffer_inn} };
  test_make_taskgraph(graph, inns);
}

void test_obvious_random_loc_matmul(int pi, int pj, int pk, int nloc) {
  graph_t graph;

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

  graph.print();

  buffer_t buffer_lhs = std::make_shared<buffer_holder_t>(ni*nj);
  buffer_lhs->iota(-10);

  buffer_t buffer_rhs = std::make_shared<buffer_holder_t>(nj*nk);
  buffer_rhs->iota(-20);

  map<int, buffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };
  test_make_taskgraph(graph, inns);
}

void test_random_matmul() {
  int nloc = 20;
  auto random_placement = [&](vector<uint64_t> total_shape) {
    vector<partdim_t> partdims;
    for(uint64_t const& n: total_shape) {
      int p = runif(1, 10);
      partdims.push_back(partdim_t::split(n, p));
    }
    partition_t part(partdims);

    return placement_t::random(part, nloc);
  };

  graph_t graph;

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

  graph.print();

  buffer_t buffer_lhs = std::make_shared<buffer_holder_t>(ni*nj);
  buffer_lhs->iota(-10);

  buffer_t buffer_rhs = std::make_shared<buffer_holder_t>(nj*nk);
  buffer_rhs->iota(-20);

  map<int, buffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };

  // TODO: uncomment this
  //test_make_taskgraph(graph, inns);

  // TODO: get this test to pass
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

void test_matmul_reference(uint64_t di, uint64_t dj, uint64_t dk) {
  buffer_t lhs = std::make_shared<buffer_holder_t>(di*dj);
  lhs->random();

  buffer_t rhs = std::make_shared<buffer_holder_t>(dj*dk);
  rhs->random();

  buffer_t rhs_true = std::make_shared<buffer_holder_t>(di*dk);
  rhs_true->zeros();
  for(int i = 0; i != di; ++i) {
  for(int j = 0; j != dj; ++j) {
  for(int k = 0; k != dk; ++k) {
    rhs_true->data[i*dk + k] += lhs->data[i*dj + j] * rhs->data[j*dk + k];
  }}}

  einsummable_t matmul = einsummable_t::from_matmul(di, dj, dk);
  buffer_t rhs_ref = reference_einsummable(matmul, {lhs, rhs});

  std::cout << rhs_true << std::endl;
  std::cout << rhs_ref  << std::endl;
}

int main(int argc, char** argv) {
//  main09(argc, argv);
  test_obvious_matmul(2,2,2);
}
