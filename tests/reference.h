#pragma once
#include "../src/einsummable/reference.h"

optional<string> test_reference_matmul() {
  uint64_t ni = 3;
  uint64_t nj = 4;
  uint64_t nk = 5;

  dtype_t dtype = dtype_t::f64;
  einsummable_t matmul = einsummable_t::from_matmul(ni,nj,nk, dtype);

  dbuffer_t lhs = make_dbuffer(dtype, ni*nj);
  dbuffer_t rhs = make_dbuffer(dtype, nj*nk);

  lhs.iota(8);
  rhs.iota(100);

  dbuffer_t out = reference_einsummable(matmul, {lhs, rhs});
  vector<double> values {
    4110.0, 4148.0, 4186.0, 4224.0, 4262.0,
    5830.0, 5884.0, 5938.0, 5992.0, 6046.0,
    7550.0, 7620.0, 7690.0, 7760.0, 7830.0
  };
  if(out.nelem() != values.size()) {
    return "invalid sized output";
  }

  for(int i = 0; i != values.size(); ++i) {
    if(!is_close(out.f64()[i], values[i])) {
      return "invalid value";
    }
  }

  return std::nullopt;
}

optional<string> test_reference_repartition() {
  partition_t partition({
    partdim_t::split(3, 1),
    partdim_t::split(4, 2),
    partdim_t::split(5, 3)
  });

  dtype_t dtype = dtype_t::f16;
  dbuffer_t tensor = make_dbuffer(dtype, 3*4*5);
  tensor.random("-0.1", "9.9");

  vtensor_t<dbuffer_t> ptensor = partition_buffer(partition, tensor);

  dbuffer_t tensor_again = unpartition_buffer(partition, ptensor);

  if(tensor != tensor_again) {
    return "Did not correctly undo the partition";
  }

  return std::nullopt;
}

optional<string> reblock_20elem_test(
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

  auto [input_gid_to_tids, output_gid_to_tids, taskgraph] = taskgraph_t::make(
    graph.graph,
    graph.get_placements());

  //taskgraph.print();

  vtensor_t<dbuffer_t> inn_pbuffer = partition_buffer(
    placement_start.partition, inn_buffer);
  vtensor_t<dbuffer_t> out_pbuffer = partition_buffer(
    placement_finish.partition, out_buffer);

  map<int, dbuffer_t> t_input_map = init_buffer_map(
    input_gid_to_tids.at(gid_inn),
    inn_pbuffer);

  auto t_out_map = typed_reference_compute_taskgraph_from_graph_info(
    taskgraph, t_input_map, graph.graph, output_gid_to_tids);

  vtensor_t<dbuffer_t> t_out_pbuffer = get_partitioned_buffer(
    t_out_map,
    output_gid_to_tids[gid_out]);
  dbuffer_t t_out_buffer = unpartition_buffer(placement_finish.partition, t_out_pbuffer);

  if(t_out_buffer != out_buffer) {
    return "reblock test failed";
  }
  return std::nullopt;
}

optional<string> reblock_20elem_test01()
{
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 5) }),
    vtensor_t<int>({5}, {0,0,0,0,0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 3) }),
    vtensor_t<int>({3}, {0,0,0})
  );

  return reblock_20elem_test(dtype_t::c64, placement_start, placement_finish);
}

optional<string> reblock_20elem_test02()
{
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 1) }),
    vtensor_t<int>({1}, {0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 2) }),
    vtensor_t<int>({2}, {0,0})
  );

  return reblock_20elem_test(dtype_t::f16, placement_start, placement_finish);
}

optional<string> reblock_20elem_test03()
{
  placement_t placement_start(
    partition_t({ partdim_t::split(20, 2) }),
    vtensor_t<int>({2}, {0,0})
  );
  placement_t placement_finish(
    partition_t({ partdim_t::split(20, 1) }),
    vtensor_t<int>({1}, {0})
  );

  return reblock_20elem_test(dtype_t::f32, placement_start, placement_finish);
}




