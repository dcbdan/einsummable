#include "misc.h"
#include "modules.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/kernels.h"
#include "../src/execution/cpu/permute.h"
#include "../src/execution/cpu/contraction.h"

#include <fstream>

#include <mkl.h> // for mkl_set_num_threads

void main_() {
  set_default_dtype(dtype_t::f16);

  auto args = model_args_t::llama_7B();

  // TODO: set vocab_size
  args.vocab_size = 45000;
  args.n_layers = 12;

  graph_writer_t writer;

  auto model = transformer_t(&writer, "name", args);

  uint64_t bsz = 8;
  uint64_t seq_len = 256;

  // read bsz and seq_len and input input tensor
  auto input_input = [&]() {
    full_shape_t shape({
      full_dim_t::singleton(bsz),
      full_dim_t::singleton(seq_len),
      args.full_dim()
    });

    return writer.input(shape);
  };

  tensor_t x = input_input();
  tensor_t y = model.forward(x);

  //seq_len = 1;

  //x = input_input();
  //y = model.forward(x);

  y = y.save();

  graph_t const& graph = writer.get_graph();

  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("wrote to g.gv");
  }

  auto const& [_0, _1, taskgraph] = taskgraph_t::make(
    graph,
    graph.make_singleton_placement());
  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("Printed to tg.gv");
  }

  DOUT("--------------");

  kernel_manager_t kernel_manager;
  int nfail = 0;
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& e = node.op.get_apply().einsummable;
      auto maybe = kernel_manager.build(e);
      if(!maybe) {
        DOUT("");
        DOUT("Could not build kernel");
        DOUT(e);
        DOUT(e.join);
        DOUT("")
        nfail ++;
      } else if(maybe.value() > 0) {
        //DOUT("Built a kernel with workspace size " << maybe.value());
      } else {
        //DOUT("Built a kernel");
      }
    }
  }
  if(nfail != 0) {
    DOUT("still " << nfail << " more kernels");
  } else {
    DOUT("all kernels are available");
  }
}

///////////////////////////////////////////////////////////////////

//void main_mm_plan() {
//  //DOUT(as_out_perm({0,1,2}, {2,0,1})); // 201
//  //DOUT(as_out_perm({1,0}, {0,1})); // 10
//  //DOUT(as_out_perm({0,1,2,3,4,5,6}, {0,1,2,3,4,6,5})); // 0123465
//  //DOUT(as_out_perm({0,1,2,3,4,6,5}, {0,1,2,3,4,5,6})); // 0123465
//  //DOUT(as_out_perm({6,1,2,3,4,0,5}, {0,1,2,3,4,5,6})); // 5123460
//  //DOUT(as_out_perm({3,4,5}, {5,3,4})); // 201
//
//  // [8,32,256,256,128]   acbe,adbe->abcd
//  //                      0214 0314  0123
//  vector<uint64_t> shape{8,32,256,256,128};
//  vector<int> lhs_inn_modes{0,2,1,4};
//  vector<int> rhs_inn_modes{0,3,1,4};
//  int out_rank = 4;
//
//  //// [8,32,256,128,256]   abce,aebd->abcd
//  ////                      0124 0413  0123
//  //vector<uint64_t> shape{8,32,256,128,256};
//  //vector<int> lhs_inn_modes{0,1,2,4};
//  //vector<int> rhs_inn_modes{0,4,1,3};
//  //int out_rank = 4;
//
//  //// [8,256,4096,32,128]  adbe,cde->abc
//  ////                      0314 234  012
//  //vector<uint64_t> shape{8,256,4096,32,128};
//  //vector<int> lhs_inn_modes{0,3,1,4};
//  //vector<int> rhs_inn_modes{2,3,4};
//  //int out_rank = 3;
//
//  vector<int> out_modes(out_rank);
//  std::iota(out_modes.begin(), out_modes.end(), 0);
//
//  auto items = enumerate_all(shape, lhs_inn_modes, rhs_inn_modes, out_rank);
//  for(auto const& [plan,workspace,cost]: items) {
//    std::cout << workspace << " | " << cost << " | ";
//    print_compute(plan, lhs_inn_modes, rhs_inn_modes, out_modes);
//    std::cout << std::endl;
//    //std::cout << " | " << workspace << " | " << cost << std::endl;
//    //DOUT(plan << " | " << workspace << " | " << cost);
//  }
//
//  //auto plan = find_smallest_workspace(shape, lhs_inn_modes, rhs_inn_modes, out_rank);
//  //DOUT(shape);
//  //print_compute(plan, lhs_inn_modes, rhs_inn_modes, out_modes);
//  //std::cout << std::endl;
//}

void time_contraction1() {
  dtype_t dtype = dtype_t::f16;

  // {8,256,4096,32,128}
  // {0,3,1,4},{2,3,4}->{0,1,2}

  {
    einsummable_t e(
      {8,256,4096,32,128},
      { {0,1,3,4}, {2,3,4} },
      3,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = build_einsummable(e.merge_adjacent_dims());
    gremlin_t timer("batch matmul " + write_with_ss(e));
    f(out_buffer.f16(), { lhs_buffer.f16(), rhs_buffer.f16() });
  }

  {
    // {8,256,4096,32,128}
    einsummable_t e(
      {8,256,4096,32,128},
      { {0,3,1,4}, {2,3,4} },
      3,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = contraction_t::make(
      dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
    buffer_t workspace = make_buffer(f.workspace_size);
    gremlin_t timer("contraction " + write_with_ss(e));
    f(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  }
}

void time_contraction2() {
  dtype_t dtype = dtype_t::f16;

  // ({0,2,1,4}->{0,1,2,4}),({0,3,1,4}->{0,1,4,3})->{0,1,2,3}
  // {8,32,256,256,128};

  {
    einsummable_t e(
      {8,32,256,256,128},
      { {0,1,2,4}, {0,1,4,3} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = build_einsummable(e.merge_adjacent_dims());
    gremlin_t timer("batch matmul " + write_with_ss(e));
    f(out_buffer.f16(), { lhs_buffer.f16(), rhs_buffer.f16() });
  }

  {
    einsummable_t e(
      {8,32,256,256,128},
      { {0,2,1,4}, {0,3,1,4} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = contraction_t::make(
      dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
    buffer_t workspace = make_buffer(f.workspace_size);
    gremlin_t timer("contraction " + write_with_ss(e));
    f(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  }
}

void time_contraction3() {
  dtype_t dtype = dtype_t::f16;

  // [8,32,256,128,256]
  // abce,aebd->abcd
  // 0124 0413  0123
  // bbij bjbk  bbik

  {
    einsummable_t e(
      {8,32,256,128,256},
      { {0,1,2,4}, {0,1,3,4} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = build_einsummable(e.merge_adjacent_dims());
    gremlin_t timer("batch matmul " + write_with_ss(e));
    f(out_buffer.f16(), { lhs_buffer.f16(), rhs_buffer.f16() });
  }

  {
    einsummable_t e(
      {8,32,256,128,256},
      { {0,1,2,4}, {0,4,1,3} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = contraction_t::make(
      dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
    buffer_t workspace = make_buffer(f.workspace_size);
    gremlin_t timer("contraction " + write_with_ss(e));
    f(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  }
}

void test_contraction1() {
  dtype_t dtype = dtype_t::f64;

  //einsummable_t e(
  //  {8,7,4,9,10},
  //  { {0,1,3,4}, {2,3,4} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);
  //einsummable_t e(
  //  {8,3,6,5,10},
  //  { {0,1,2,4}, {0,1,4,3} },
  //  4,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);
  einsummable_t e(
    {8,3,6,5,10},
    { {0,4,2,1}, {0,1,4,3} },
    4,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  auto inn_shapes = e.inn_shapes();
  dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
  dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));

  lhs_buffer.random();
  rhs_buffer.random();

  dbuffer_t out_buffer_true =
    reference_einsummable(e, {lhs_buffer, rhs_buffer});

  dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

  auto contraction = contraction_t::make(
    dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
  buffer_t workspace = make_buffer(contraction.workspace_size);

  out_buffer.random();
  contraction(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  if(is_close(out_buffer, out_buffer_true)) {
    DOUT("yes, is close");
  } else  {
    DOUT("IS NOT CLOSE!");
  }
}

int main() {
  main_();

  //main_mm_plan();

  //test_contraction1();

  //mkl_set_num_threads(1);
  //time_contraction1();
  //DOUT("--------");
  //time_contraction2();
  //DOUT("--------");
  //time_contraction3();
}
