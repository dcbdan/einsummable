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

int main() {
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 5) }),
    tensor_t<int>({5}, {0,0,0,0,0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 5) }),
    tensor_t<int>({5}, {0,0,0,0,0})
  );

  buffer_t inn_buffer = std::make_shared<buffer_holder_t>(20);
  inn_buffer->iota(0);

  graph_t graph;
  {
    int inn = graph.insert_input(placement_start);
    graph.insert_formation(placement_finish, inn);
  }

  auto [input_map, output_map, taskgraph] = taskgraph_t::make(graph);

  taskgraph.print();
}
