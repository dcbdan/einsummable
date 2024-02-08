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

taskgraph_t test_make_taskgraph(
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<int, dbuffer_t> full_inns)
{
  tuple<
    map<int, vtensor_t<int> >,
    map<int, vtensor_t<int> >,
    taskgraph_t>
    _info = taskgraph_t::make(graph, placements);
  auto const& [inn_to_blocks, out_to_blocks, taskgraph] = _info;

  //{
  //  std::cout << "Printing to tg.gv" << std::endl;
  //  std::ofstream f("tg.gv");
  //  taskgraph.print_graphviz(f);
  //}

  //taskgraph.print();

  map<int, dbuffer_t> task_inns;
  for(auto [gid, full_buffer]: full_inns) {
    vtensor_t<dbuffer_t> pbuffer = partition_buffer(
      placements.at(gid).partition,
      full_buffer);
    fill_buffer_map(task_inns, inn_to_blocks.at(gid), pbuffer);
  }

  map<int, dbuffer_t> full_outs = reference_compute_graph(graph, full_inns);

  map<int, dbuffer_t> task_outs =
    typed_reference_compute_taskgraph_from_graph_info(
      taskgraph, task_inns, graph, out_to_blocks);

  for(auto const& [gid, full_buffer_via_graph]: full_outs) {
    vtensor_t<dbuffer_t> t_part_buffer =
      get_partitioned_buffer(task_outs, out_to_blocks.at(gid));
    vtensor_t<dbuffer_t> part_buffer =
      partition_buffer(placements.at(gid).partition, full_buffer_via_graph);

    auto const& tids  = out_to_blocks.at(gid).get();
    auto const& t_vec = t_part_buffer.get();
    auto const& vec   = part_buffer.get();
    for(int i = 0; i != vec.size(); ++i) {
      auto const& tid = tids[i];
      auto const& t   = t_vec[i];
      auto const& u   = vec[i];
      if(!is_close(t, u)) {
        //std::cout << tid << std::endl << t << std::endl << u << std::endl;
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
  return taskgraph;
}

taskgraph_t test_make_taskgraph(
  graph_constructor_t const& graph,
  map<int, dbuffer_t> full_inns)
{
  return test_make_taskgraph(graph.graph, graph.get_placements(), full_inns);
}


void test_make_memgraph_with_evict(
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<int, dbuffer_t> full_inns,
  vector<int> const& which_storage,
  vector<uint64_t> mem_sizes)
{
  // graph.print();
  // for(int i = 0; i != graph.nodes.size(); ++i) {
  //  DOUT(i << ": " << placements[i].partition);
  // }

  tuple<
    map<int, vtensor_t<int> >,
    map<int, vtensor_t<int> >,
    taskgraph_t>
    _info0 = taskgraph_t::make(graph, placements);
  auto const& [inn_to_blocks, out_to_blocks, taskgraph] = _info0;

  {
   std::cout << "Printing to exp_reference_taskgraph.gv" << std::endl;
   std::ofstream f("exp_reference_taskgraph.gv");
   taskgraph.print_graphviz(f);
  }

  int num_locs = taskgraph.num_locs();

  tuple<
    map<int, memstoloc_t>, // input -> data
    map<int, memstoloc_t>, // save  -> data
    memgraph_t>
    _info1 = memgraph_t::make(taskgraph, which_storage, mem_sizes);
  auto const& [task_inn_to_mem, task_out_to_mem, memgraph] = _info1;

  DOUT("After memgraph make...");

  // verify that deduced mem_sizes match with provided mem sizes
  {
    auto mem_sizes_ = memgraph.mem_sizes();
    if(mem_sizes.size() != mem_sizes_.size()) {
      throw std::runtime_error("mem_sizes invalid");
    }
    for(int i = 0; i != mem_sizes.size(); ++i) {
      if(mem_sizes_[i] > mem_sizes[i]) {
        throw std::runtime_error("memory allocated past provided mem_sizes");
      }
    }
  }

  // allocate a blob of memory at each compute location
  vector<buffer_t> loc_buffers;
  for(uint64_t const& mem_sz: mem_sizes) {
    loc_buffers.push_back(std::make_shared<buffer_holder_t>(mem_sz));
  }

  // Declare a vector that stores buffer in each storage location
  vector<map<int, buffer_t>> sto_buffers(num_locs);

  // Initialize the taskgraph inputs and then
  // copy into the location wise buffers
  {
    map<int, dbuffer_t> task_inns;
    for(auto [gid, full_buffer]: full_inns) {
      vtensor_t<dbuffer_t> pbuffer = partition_buffer(
        placements.at(gid).partition,
        full_buffer);
      fill_buffer_map(task_inns, inn_to_blocks.at(gid), pbuffer);
    }

    for(auto const& [tid, dbuffer]: task_inns) {
      memstoloc_t inn_memstoloc = task_inn_to_mem.at(tid);
      if (inn_memstoloc.is_stoloc()) {
        auto const& [sto_loc, sto_id] = inn_memstoloc.get_stoloc();
        sto_buffers[sto_loc][sto_id] = dbuffer.data;
      } else if (inn_memstoloc.is_memloc()) {
        auto buffer = dbuffer.data;
        auto const& [offset, size, loc] = inn_memstoloc.get_memloc();
        if(size != buffer->size) {
          throw std::runtime_error("maybe invalid task_inn_to_mem");
        }
        std::copy(
          buffer->data, buffer->data + size,
          loc_buffers[loc]->data + offset);
      }
    }
  }

  // compute the reference implementation
  map<int, dbuffer_t> full_outs = reference_compute_graph(graph, full_inns);

  {
   std::cout << "Printing to exp_reference_memgraph.gv" << std::endl;
   std::ofstream f("exp_reference_memgraph.gv");
   memgraph.print_graphviz(f);
  }

  reference_compute_memgraph(memgraph, loc_buffers, sto_buffers);


  for(auto const& [gid, full_buffer]: full_outs) {
    vtensor_t<dbuffer_t> part_buffer =
      partition_buffer(placements.at(gid).partition, full_buffer);

    auto const& tids = out_to_blocks.at(gid).get();
    auto const& vec  = part_buffer.get();
    for(int i = 0; i != vec.size(); ++i) {
      int       const& tid = tids[i];
      dbuffer_t const& from_graph = vec[i];

      // where in the loc_buffers/sto_buffers the result is
      int loc = taskgraph.nodes[tid].op.out_loc();
      memstoloc_t out_memstoloc = task_out_to_mem.at(tid);
      dbuffer_t from_memgraph;
      if (out_memstoloc.is_memloc()) {
        auto const& [offset, size, loc] = out_memstoloc.get_memloc();
        from_memgraph = dbuffer_t(
          from_graph.dtype,
          make_buffer_reference(loc_buffers[loc]->data + offset, size));
      } else {
        auto const& [sto_loc, sto_id] = out_memstoloc.get_stoloc();
        from_memgraph = dbuffer_t(
          from_graph.dtype,
          make_buffer_reference(sto_buffers[sto_loc].at(sto_id)->data, sto_buffers[sto_loc].at(sto_id)->size));
      }
      if(!is_close(from_graph, from_memgraph)) {
        std::cout << "expected: " << from_graph    << std::endl;
        std::cout << "actual:   " << from_memgraph << std::endl;
        throw std::runtime_error("make memgraph without evict test fail");
      }
    }
  }

}

void test_make_memgraph_without_evict(
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<int, dbuffer_t> full_inns)
{
  //graph.print();
  //for(int i = 0; i != graph.nodes.size(); ++i) {
  //  DOUT(i << ": " << placements[i].partition);
  //}

  tuple<
    map<int, vtensor_t<int> >,
    map<int, vtensor_t<int> >,
    taskgraph_t>
    _info0 = taskgraph_t::make(graph, placements);
  auto const& [inn_to_blocks, out_to_blocks, taskgraph] = _info0;

  //{
  //  std::cout << "Printing to exp_reference_taskgraph.gv" << std::endl;
  //  std::ofstream f("exp_reference_taskgraph.gv");
  //  taskgraph.print_graphviz(f);
  //}

  int num_locs = taskgraph.num_locs();

  tuple<
    map<int, mem_t>, // input -> mem
    map<int, mem_t>, // save -> mem
    memgraph_t>
    _info1 = memgraph_t::make_without_evict(taskgraph);
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
      vtensor_t<dbuffer_t> pbuffer = partition_buffer(
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
    vtensor_t<dbuffer_t> part_buffer =
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

void test_obvious_matmul_with_evict(int pi, int pj, int pk,
  uint64_t ni = 20, uint64_t nj = 20, uint64_t nk = 20,
  uint64_t mem_size = 2000)
{
  graph_constructor_t graph;

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
  buffer_lhs.random("-0.001", "0.002");

  dbuffer_t buffer_rhs = make_dbuffer(default_dtype(), nj*nk);
  buffer_rhs.random("-0.001", "0.001");

  map<int, dbuffer_t> inns{ {id_lhs, buffer_lhs}, {id_rhs, buffer_rhs} };

  int n_compute_locs = 0;
  vector<placement_t> placements = graph.get_placements();
  for (auto const& placement: placements) {
    for (auto const& location: placement.locations.get()) {
      if (location >= n_compute_locs) {
        n_compute_locs = location + 1;
      }
    }
  }

  vector<int> which_storage(n_compute_locs);
  std::iota(which_storage.begin(), which_storage.end(), 0);

  vector<uint64_t> mem_sizes(n_compute_locs, mem_size);

  test_make_memgraph_with_evict(graph.graph, graph.get_placements(), inns, which_storage, mem_sizes);
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
  for(int seed = 0; seed != 100; ++seed) {
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
  buffer0.random("-0.001", "0.001");
  //buffer0.iota(-10);

  dbuffer_t buffer1 = make_dbuffer(default_dtype(), nj*nk);
  buffer1.random("-0.001", "0.001");
  //buffer1.iota(-20);

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

taskgraph_t test_random_concat(
  int dim,
  vector<uint64_t> shape_template,
  int n_inn,
  optional<dtype_t> maybe_dtype = std::nullopt,
  int maxpart = 10)
{
  dtype_t dtype;
  if(maybe_dtype) {
    dtype = maybe_dtype.value();
  } else {
    dtype = dtype_random();
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

  x = x.save();

  int nloc = 3;
  random_placement_t random_placement { {1, maxpart}, nloc };

  graph_t g = w.get_graph();
  vector<placement_t> pls;
  for(int gid = 0; gid != g.nodes.size(); ++gid) {
    pls.push_back(random_placement(g.nodes[gid].op.shape()));
  }

  return test_make_taskgraph(g, pls, inn_tensors);
}

void main13() {
  graph_writer_t w;

  using id_t = graph_writer_t::tensor_t;

  dtype_t dtype = dtype_t::f64;

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

  x = x.save();
  y = y.save();

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
    // Now make sure the output formation nodes have the same placement
    for(int id = 0; id != g.nodes.size(); ++id) {
      auto const& node = g.nodes[id];
      if(node.op.is_formation()) {
        int inn_id = node.inns[0];
        int inn_rank = pls[inn_id].partition.block_shape().size();
        int rank = pls[id].partition.block_shape().size();
        if(inn_rank == rank) {
          pls[id] = pls[inn_id];
        }
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

void main15(int argc, char** argv) {
  if(argc != 4) {
    throw std::runtime_error("usage: pi pj pk");
  }
  int pi = parse_with_ss<int>(argv[1]);
  int pj = parse_with_ss<int>(argv[2]);
  int pk = parse_with_ss<int>(argv[3]);
  test_obvious_matmul_with_evict(pi, pj, pk);
}

void test_random_goofy_ff() {
  graph_writer_t writer;
  using id_t = graph_writer_t::tensor_t;

  uint64_t bsz = 3;
  uint64_t d0 = 4;
  uint64_t d1 = 5;
  uint64_t d2 = 6;
  uint64_t d3 = 8;

  vector<uint64_t> d01 = {d0, d1};
  vector<uint64_t> d12 = {d1, d2};
  vector<uint64_t> d22 = {d2, d2};
  vector<uint64_t> d32 = {d3, d2};
  vector<uint64_t> d33 = {d3, d3};

  dtype_t dtype = dtype_t::f32;

  id_t x = writer.input({bsz,d0,d1}, dtype).view({ {bsz}, d01 });
  id_t w0 = writer.input({d0,d1,d3,d2}, dtype).view({d01,d32});
  id_t w1 = writer.input({d3,d2}, dtype);

  map<int,dbuffer_t> inns;
  for(id_t id: vector<id_t>{x,w0,w1}) {
    int gid = id.get_id();
    dbuffer_t buffer = make_dbuffer(dtype, product(id.get_shape()()));
    buffer.random();
    inns.insert({gid, buffer});
  }

  id_t y = writer.matmul(x, w0).view_full({bsz, d3, d2});
  id_t z = writer.matmul(y, w1.transpose(0,1)); // bsz,d3,d3
  y = writer.concat(2, {y, z}); // bsz,d3,d33
  y = writer.reduction("bxy->bx", castable_t::add, y); // bsz,d1
  y = y.to_complex();
  y = y.to_real();

  y = y.save();

  int nloc = 3;
  random_placement_t random_placement { {1, 10}, nloc };

  graph_t g = writer.get_graph();
  vector<placement_t> pls;
  for(int gid = 0; gid != g.nodes.size(); ++gid) {
    pls.push_back(random_placement(g.nodes[gid].op.shape()));
  }

  test_make_taskgraph(g, pls, inns);
}

void test_with_complex_matmul() {
  graph_writer_t writer;
  using id_t = graph_writer_t::tensor_t;

  uint64_t ni = 10;
  uint64_t nj = 12;
  uint64_t nk = 14;

  id_t lhs = writer.input({ni,nj}, dtype_t::c64);
  id_t rhs = writer.input({nj,nk}, dtype_t::c64);
  id_t out = writer.matmul(lhs, rhs);

  vector<int> sames;
  sames.push_back(out.get_id());
  out = out.to_real();
  sames.push_back(out.get_id());
  out = out.to_complex();
  sames.push_back(out.get_id());
  out = out.to_real();
  sames.push_back(out.get_id());
  out = writer.add(out, out);
  sames.push_back(out.get_id());
  out = out.to_complex();
  sames.push_back(out.get_id());
  out = out.to_real();
  sames.push_back(out.get_id());
  out = out.to_complex();
  sames.push_back(out.get_id());
  out = writer.add(out, out);
  sames.push_back(out.get_id());

  out = out.save();

  map<int,dbuffer_t> inns;
  for(id_t id: vector<id_t>{lhs, rhs}) {
    int gid = id.get_id();
    dbuffer_t buffer = make_dbuffer(dtype_t::c64, product(id.get_shape()()));
    buffer.random();
    inns.insert({gid, buffer});
  }

  graph_t const& graph = writer.get_graph();

  //{
  //  vector<placement_t> placements = graph.make_singleton_placement(graph);
  //  test_make_taskgraph(graph, placements, inns);

  //  placements = vector<placement_t>();
  //  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
  //    auto const& node = graph.nodes[gid];
  //    auto shape = node.op.shape();
  //    vector<partdim_t> partdims;
  //    for(auto sz: shape) {
  //      partdims.push_back(partdim_t::split(sz, 2));
  //    }
  //    placements.emplace_back(partition_t(partdims));
  //  }

  //  test_make_taskgraph(graph, placements, inns);
  //}

  {
    int nloc = 3;
    random_placement_t random_placement { {1, 4}, nloc };
    graph_t const& g = graph;
    vector<placement_t> pls;
    for(int gid = 0; gid != g.nodes.size(); ++gid) {
      pls.push_back(random_placement(g.nodes[gid].op.shape()));
    }

    // This was used to verify that to_complex and to_real get
    // correctly compiled out
    //
    //for(auto const& id: sames) {
    //  if(dtype_is_real(graph.nodes[id].op.out_dtype())) {
    //    pls[id] = double_last_dim(pls[sames[0]]);
    //  } else {
    //    pls[id] = pls[sames[0]];
    //  }
    //}

    test_make_taskgraph(g, pls, inns);
    test_make_memgraph_without_evict(graph, pls, inns);
  }
}

void main_pass_through_partials_stuff() {
  {
    taskgraph_t tg;
    dtype_t dtype = default_dtype();

    int a = tg.insert_input(0, dtype, {100});
    int b = tg.insert_input(0, dtype, {100});

    int x = tg.new_partial(0, dtype, {100});
    tg.add_to_partial_the_full_aggregate(x, a, castable_t::add, false);

    int y = tg.new_partial(0, dtype, {100});
    tg.add_to_partial_the_full_aggregate(y, x, castable_t::add, false);
    tg.add_to_partial_the_full_aggregate(y, b, castable_t::add, false);
    tg.nodes[y].is_save = true;

    DOUT("Printing the passthrough partials: ");
    DOUT("  (x id is " << x << ")");
    for(auto const& id: tg.collect_passthrough_partials()) {
      DOUT("  id " << id);
    }
  }

  {
    taskgraph_t tg;
    dtype_t dtype = default_dtype();

    int a1 = tg.insert_input(0, dtype, {50});
    int a2 = tg.insert_input(0, dtype, {50});
    int b  = tg.insert_input(0, dtype, {100});

    int x = tg.new_partial(0, dtype, {100});
    tg.add_to_partial(x, a1,
      touch_t {
        .selection = { { 50, 100, 0, 0, 50 } },
        .castable = std::nullopt,
        .dtype = dtype
      },
      false);
    tg.add_to_partial(x, a2,
      touch_t {
        .selection = { { 50, 100, 0, 50, 50 } },
        .castable = std::nullopt,
        .dtype = dtype
      },
      false);

    int y = tg.new_partial(0, dtype, {100});
    tg.nodes[y].is_save = true;
    tg.add_to_partial_the_full_aggregate(y, x, castable_t::add, false);
    tg.add_to_partial_the_full_aggregate(y, b, castable_t::add, false);

    DOUT("Printing the passthrough partials: ");
    DOUT("  (x id is " << x << ")");
    for(auto const& id: tg.collect_passthrough_partials()) {
      DOUT("  id " << id);
    }
  }
}

void main_touch_compose() {
  auto make_1d_touch = [](vector<uint64_t> v) {
    return touch_t {
      .selection = { touchdim_t {
        v[0], v[1], v[2], v[3], v[4]
      }},
      .castable = std::nullopt,
      .dtype = default_dtype()
    };
  };

  touch_t a = make_1d_touch({4,6,0,1,2});
  touch_t b = make_1d_touch({6,4,2,0,1});
  DOUT(touch_compose(a,b).value());
}

void main_subset(int which) {
  dtype_t dtype = dtype_random();

  graph_writer_t writer;

  using id_t = graph_writer_t::tensor_t;

  using _all = graph_writer_t::idx_t::all;
  using _rng = graph_writer_t::idx_t::rng;
  using _idx = graph_writer_t::idx_t::idx;

  id_t x = writer.input({{30,8}, {5}}, dtype).view_full();

  if(which == 0) {
    // Don't use all of X
    id_t y = x.subset({ _all{}, _idx{-1}, _all{} });
    id_t z = x.subset({ _all{}, _rng{2,4}, _rng{0,4} });

    y = y.save();
    z = z.save();
  } else if(which == 1) {
    // Use all of X
    id_t y = x.subset({ _all{}, _idx{-1}, _all{} });
    id_t z = writer.add(x, x);
    id_t w = writer.ew("ijk,ik->ijk", scalarop_t::make_mul(dtype), x, y);

    y = y.save();
    z = z.save();
    w = w.save();
  } else if(which == 2) {
    scalar_t scalar =
      dtype_is_complex(dtype)                         ?
      scalar_t(std::complex<float>(0.01, 0.01)) :
      scalar_t(dtype, "0.01");

    id_t y = x.subset({ _rng{0, 5}, _rng{0, 5}, _all{} });
    id_t z = y.scale(scalar);

    z = z.save();
  } else if(which == 3) {
    id_t y = x.view({{30,8}, {5}});
    id_t z = y.subset({ _all{}, _rng{0,2} });
    z = z.save();
  } else {
    throw std::runtime_error("main subset invalid which");
  }

  dbuffer_t x_data = make_dbuffer(dtype, product(x.get_shape()()));
  x_data.random();

  graph_t const& graph = writer.get_graph();

  int nloc = 4;
  random_placement_t random_placement { {1, 5}, nloc };

  vector<placement_t> pls;
  for(auto const& node: graph.nodes) {
    pls.push_back(random_placement(node.op.shape()));
  }

  test_make_taskgraph(
    graph,
    pls,
    { {x.get_id(), x_data} });
}

void mm_test() {
  int np = 2;
  auto g = three_dimensional_matrix_multiplication(2, 2, 2, 200, 200, 200, np);
  auto graph = g.graph;
  auto pls = g.get_placements();

  auto [_2, _3, taskgraph] = taskgraph_t::make(graph, pls);

  uint64_t mem_size = 0.001 * 1024lu * 1024lu * 1024lu;

  bool split_off_inputs = true;
  auto [_0, _1, maybe_mg, exec_mg] = memgraph_t::make_(
    taskgraph, {}, vector<uint64_t>(np, mem_size), {},
    allocator_settings_t::default_settings(),
    true, split_off_inputs);

  if(maybe_mg) {
    std::ofstream f("i_mg.gv");
    maybe_mg.value().print_graphviz(f);
    DOUT("i_mg.gv");
  }

  std::ofstream f("e_mg.gv");
  exec_mg.print_graphviz(f);
  DOUT("e_mg.gv");
}

int main(int argc, char** argv) {
  //test_random_matmul();

  //main09(argc, argv);
  // main10();
  //main11(argc, argv);
  //set_seed(0);
  //test_obvious_random_loc_matmul(5,5,5,5);

  // for(int i = 0; i != 10; ++i) {
  //  test_random_matmul_then_unary_ew(scalarop_t::make_increment(scalar_t(float(0.77))));
  //  DOUT(i+1);
  // }

  //main13();
  //main14();

  // Example 1
  //set_seed(0);
  //test_random_concat(2, {20,19,18}, 3, std::nullopt, 3);

  // Example 2
  //set_seed(0);
  //test_random_concat(0, {20,18}, 2, std::nullopt, 3);

  //for(int i = 0; i != 1000; ++i) {
  //  DOUT(i);
  //  set_seed(i);
  //  test_random_goofy_ff();
  //}

  //set_seed(0);
  //test_random_goofy_ff();

  //main02();
  //main03();
  //main04();
  //main05();

  //for(int i = 0; i != 100; ++i) {
  //  DOUT(i);
  //  set_seed(i);
  //  test_with_complex_matmul();
  //}

  // for(int i = 0; i != 100; ++i) {
  //   DOUT(i);
  //   set_seed(i);
  //   for(int which = 0; which != 4; ++which) {
  //     main_subset(which);
  //   }
  // }
  // main06(argc, argv);
  //main08(argc, argv);

  //test_obvious_matmul_with_evict(2, 2, 2, 40, 40, 40, 20*20*4*3 * 2);

  mm_test();
}


