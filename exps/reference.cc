#include "../src/reference.h"

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

// TODO: assert they are correct
// TODO: rename repartition_test
void exp_process01(
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
}

void main05() {
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 5) }),
    tensor_t<int>({5}, {0,0,0,0,0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 3) }),
    tensor_t<int>({3}, {0,0,0})
  );

  exp_process01(placement_start, placement_finish);
}

void main06() {
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 1) }),
    tensor_t<int>({1}, {0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 2) }),
    tensor_t<int>({2}, {0,0})
  );

  exp_process01(placement_start, placement_finish);
}

void main07() {
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 2) }),
    tensor_t<int>({2}, {0,0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 1) }),
    tensor_t<int>({1}, {0})
  );

  exp_process01(placement_start, placement_finish);
}

int main() {
  main05();
}


