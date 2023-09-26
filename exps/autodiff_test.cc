#include "../src/einsummable/graph.h"
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/memgraph.h"
#include <fstream>
#include "../src/base/setup.h"

void scale_derivative() {

  graph_constructor_t gc;

  int i1 = gc.insert_input(
    {2, 3, 4, 5}
  );

  scalar_t factor(5.0);

  einsummable_t scale({2, 3, 4, 5},
  {{0,1,2,3}},
  4,
  scalarop_t::make_scale(factor)
  );

  int s1 = gc.insert_einsummable(scale, {i1});

  gc.graph.print();
}

void mul_derivative() {
  
  graph_t graph; 

  vector<uint64_t> input_shape1 = {4, 5, 2, 3};
  vector<uint64_t> input_shape2 = {4, 5, 3, 5};
  vector<uint64_t> input_shape3 = {4, 5, 5, 2};
  vector<uint64_t> input_shape4 = {4, 5, 2, 7};
  vector<uint64_t> input_shape5 = {2, 4, 5, 7};

  int i1 = graph.insert_input(
    input_shape1
  );

  int i2 = graph.insert_input(
    input_shape2
  );

  int i3 = graph.insert_input(
    input_shape3
  );

  int i4 = graph.insert_input(
    input_shape4
  );

  auto const& notation  = einsummable_t::create_batch_matmul_string(4, 4, false, false);
  auto const& [inns, out_rank] = einsummable_t::parse_str(notation);

  auto join_shape1 = einsummable_t::construct_join_shape(inns, {input_shape1, input_shape2}).value();
  auto join_shape2 = einsummable_t::construct_join_shape(inns, {input_shape3, input_shape4}).value();

  einsummable_t matmul1(
    join_shape1,
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  );

  einsummable_t matmul2(
    join_shape2,
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  ); 

  int m1 = graph.insert_einsummable(
    matmul1, 
    {i1, i2}
  );

  int m2 = graph.insert_einsummable(
    matmul2,
    {i3, i4}
  );

  auto const& lhs = graph.nodes[m1];
  auto const& rhs = graph.nodes[m2];

  auto join_shape3 = einsummable_t::construct_join_shape(inns, {lhs.op.shape(), rhs.op.shape()}).value();

  einsummable_t matmul3(
    join_shape3,
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  ); 

  int m3 = graph.insert_einsummable(
    matmul3,
    {m1, m2}
  );

  auto [inns1, out_ranked] = einsummable_t::parse_str("abcd->acdb");

  std::cout << inns1 << std::endl;

  std::cout << "SHAPE" << std::endl;
  std::cout << matmul3.out_shape() << std::endl;
  auto mat = forward_permute(inns1[0], matmul3.out_shape());
  std::cout << mat << std::endl;
  std::cout << backward_permute(inns1[0], mat) << std::endl;
  auto const& join_shape4 = einsummable_t::construct_join_shape({inns1}, {matmul3.out_shape()});

  std::cout << "Join shape: " << join_shape4.value() << std::endl;

  einsummable_t ewu(
    join_shape4.value(),
    inns1,
    out_ranked,
    scalarop_t::make_relu()
  );

  std::cout << ewu.str() << std::endl;

  int u1 = graph.insert_einsummable(
    ewu,
    {m3}
  );

  int i5 = graph.insert_input(
    input_shape5
  );

  auto [inns2, out_rank2] = einsummable_t::parse_str("abcd,badc->abcd");
  auto const& join_shape5 = einsummable_t::construct_join_shape(inns2, {ewu.out_shape(), input_shape5});
  string ds = "f32";

  einsummable_t ewb(
    join_shape5.value(),
    inns2, 
    out_rank2,
    scalarop_t::from_string(
      "power{2}[+[*[hole|"+ds+"@0,constant{"+ds+"|5}],*[hole|"+ds+"@1,constant{"+ds+"|-5}]]]")
  );

  int b1 = graph.insert_einsummable(ewb, {u1, i5});

  auto [inns3, out_rank3] = einsummable_t::parse_str("abcd->abc");
  auto const& join_shape6 = einsummable_t::construct_join_shape(inns3, {ewb.out_shape()});

  einsummable_t reduction(
    join_shape6.value(),
    inns3,
    out_rank3,
    scalarop_t::make_identity(),
    castable_t::max
  );

  int r1 = graph.insert_einsummable(reduction, {b1});

  graph.backprop({r1}, {i1, i3});

  graph.print();
  
  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("wrote g.gv");
  }
}

void permute_test() {

  // vector<int> inn1 = {0, 3, 1, 2};
  // std::cout << inverse_permute(inn1) << std::endl;

  string e_str1 = "bacd,dcab->abcd";
  vector<uint64_t> input1 = {3, 2, 5, 7};
  vector<uint64_t> input2 = {7, 5, 2, 3};
  std::cout << input1 << " " << input2 << std::endl;
  vector<vector<uint64_t>> inn_shapes = {input1, input2};
  auto const& [inns, out_rank] = einsummable_t::parse_str(e_str1);
  std::cout << inns << " " << out_rank << std::endl;
  auto join_shape = einsummable_t::construct_join_shape(inns, {input1, input2}).value(); 

  std::cout << "Join shape: " << join_shape << std::endl;

  einsummable_t einsum(
    join_shape, 
    inns, 
    out_rank,
    scalarop_t::make_add()
  );

  std::cout << einsum << std::endl;
}

vector<int> find_path(vector<uint64_t> input, vector<uint64_t> output) {

  set<uint64_t> input_set(input.begin(), input.end());
  set<uint64_t> output_set(output.begin(), output.end());

  if (input_set.size() != output_set.size()) {
    std::runtime_error("Input and output must be the same size");
  }

  if (input_set != output_set) {
    std::runtime_error("Input and output sets are not equal");
  }

  map<int, int> output_map; 
  vector<int> ret;

  for (int i = 0; i < output.size(); i++) {
    output_map.insert({output[i], i});
  }

  for (int i = 0; i < input.size(); i++) {
    ret.push_back(output_map[input[i]]);
  }

  return ret;
}

void test_VJP() {

  vector<uint64_t> o1 = {2, 3, 5}; // 235, 325 -> 253  --------- acb, bac
  vector<uint64_t> o2 = {3, 2, 5}; 
  auto [inns, out_rank] = einsummable_t::parse_str("acb,cab->abc");
  auto join_rank = einsummable_t::construct_join_shape(inns, {o1, o2}).value();

  einsummable_t einsum(
    join_rank,
    inns, 
    out_rank,
    scalarop_t::make_add()
  );

  std::cout << join_rank << std::endl;

  // Suppose that VJP of this einsummable node wrt to o2 is dependent only on o1 (Assume that is ew mul) 
  auto join_rank_grad = einsummable_t::construct_join_shape({{find_permutation(einsum.inns[0], einsum.inns[1])}, find_permutation({0,1,2}, einsum.inns[1])}, {o1, einsum.out_shape()}).value();


  std::cout << einsummable_t::create_binary_vjp_string(einsum.inns[0], einsum.inns[0]) << std::endl;


  std::cout << join_rank_grad << std::endl;

  // vector<int> innp = {0, 1, 2};
  // vector<int> innp2 = {0, 2, 1};
  // vector<int> outp = {2, 0, 1};
  // std::cout << as_out_perm({0, 1, 2}, {2, 0, 1}) << std::endl; // Expected: {1, 2, 0}
  // std::cout << as_out_perm({0, 2, 1}, {2, 0, 1}) << std::endl; // Expected: {1, 0 ,2}
  // std::cout << "FIND PERMUTATION FOR: " << find_permutation(innp, outp) << std::endl;
  // std::cout << "FIND PERMUTATION FOR: " << find_permutation(innp2, outp) << std::endl;

}

void test_mm() {

  vector<uint64_t> input_shape1 = {4, 5, 2, 3};
  vector<uint64_t> input_shape2 = {4, 5, 3, 5};

  graph_t graph; 

  graph.insert_input(
    input_shape1
  );

  graph.insert_input(
    input_shape2
  );

  auto const& notation  = einsummable_t::create_batch_matmul_string(4, 4, false, false);
  auto const& [inns, out_rank] = einsummable_t::parse_str(notation);

  auto join_shape1 = einsummable_t::construct_join_shape(inns, {input_shape1, input_shape2}).value();

  einsummable_t mm(
    join_shape1,
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  );

  std::cout << mm.join_shape << std::endl;
  std::cout << mm.out_shape() << std::endl;

  std::cout << mm.inn_shapes() << std::endl;
}

string get_deri_einsum(einsummable_t einsum, int which_inn) {

  map<int, int> mapz;
  vector<vector<int>> inns;
  vector<int> tmp;
  int ctr = 0;

  for (auto i : einsum.inns[which_inn]) {
    mapz.insert({i, ctr++});
  }

  vector<int> tmp2;

  for (auto i : einsum.inns[which_inn == 0 ? 1 : 0]) {
    auto it = mapz.find(i);
    if (it == mapz.end()) {
      mapz.insert({i, ctr});
      tmp2.push_back(ctr++);
    } else {
      tmp2.push_back(it->second);
    }
  }

  std::cout << tmp2 << std::endl;

  for (int i = 0; i < einsum.out_rank; i++) {
    tmp.push_back(mapz.at(i));
  }

  std::cout << tmp << std::endl;

  inns.push_back(tmp);
  inns.push_back(tmp2);

  std::cout << inns << std::endl;

  auto stringy = einsummable_t::make_str(inns, einsum.inns[which_inn].size());
  std::cout << stringy << std::endl;

  return stringy;
}

void contraction_test() {

  vector<uint64_t> input_shape1 = {4, 5, 2, 3};
  vector<uint64_t> input_shape2 = {4, 5, 3, 5};
  vector<uint64_t> input_shape3 = {6, 7};

  graph_t graph; 

  graph.insert_input(
    input_shape1
  );

  graph.insert_input(
    input_shape3
  );

  string einsum = "ijkl,mn->ijmn";
  auto const& [inns, out_rank] = einsummable_t::parse_str(einsum);
  auto const& join_shape = einsummable_t::construct_join_shape(inns, {input_shape1, input_shape3});

  einsummable_t cc(
    join_shape.value(),
    inns, 
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  );

  std::cout << cc.join_shape << std::endl;
  std::cout << cc.out_shape() << std::endl;
  std::cout << cc.inn_shapes() << std::endl;
  std::cout << cc.inns << std::endl;
  std::cout << cc.get_input_from_join(join_shape.value(), 0) << std::endl;
  std::cout << cc.get_input_from_join(join_shape.value(), 1) << std::endl;

  auto const& [deri_inns, deri_out_rank] = einsummable_t::parse_str("ijmn,mn->ijkl");
  std::cout << deri_inns << std::endl;

  //{4,6,7,8},{7,5,2,8}

  auto const& [mm_inns, mm_or] = einsummable_t::parse_str("ijk,kl->il");
  auto const& join_shape_mm = einsummable_t::construct_join_shape(mm_inns, {{4,5,5},{5,7}});
  std::cout << join_shape_mm.value() << std::endl;
  einsummable_t mm(
    join_shape_mm.value(),
    mm_inns,
    mm_or,
    scalarop_t::make_mul(),
    castable_t::add
  );

  if (mm.is_contraction()) {
    std::cout << "contract" << std::endl;
  }

  std::cout << join_shape_mm.value() << std::endl;

  auto const& terms = einsummable_t::make_str_terms(mm.inns, mm.out_rank);
  auto const& stringovic = std::get<1>(terms)[0] + "," + std::get<0>(terms) + "->" + std::get<1>(terms)[1];
  //auto const& stringovic = get_deri_einsum(mm, 0);
 
  std::cout << einsummable_t::normalize_str(stringovic) << std::endl; 

  auto const& [dinns, dor] = einsummable_t::parse_str(stringovic); 
  //auto const& djs = einsummable_t::construct_join_shape(dinns, {mm.out_shape(), mm.inn_shapes()[1]});
  std::cout << mm.out_shape() << "," << mm.inn_shapes()[0] << "->" << mm.inn_shapes()[1] << std::endl;
  //std::cout << djs.value() << std::endl;
  std::cout << dinns << " " << dor << std::endl;
  auto const& new_join_shape = contraction_remap(mm.inn_shapes()[1], mm.join_shape);

  einsummable_t derija(
    new_join_shape,
    dinns,
    dor,
    scalarop_t::make_mul(),
    castable_t::add
  );

  std::cout << "AJDEEE" << std::endl;
  std::cout << derija << std::endl;
}

void reduction_test() {



  vector<uint64_t> input_shape3 = {6, 7};
  auto const& [inns, out_rank] = einsummable_t::parse_str("ij->i");
  auto const& join_shape = einsummable_t::construct_join_shape(inns, {input_shape3});

  einsummable_t reduction_sum(
    join_shape.value(),
    inns,
    out_rank,
    scalarop_t::make_identity(),
    castable_t::add
  );

  auto const terms = einsummable_t::make_str_terms(inns, out_rank);

  string vjp_string = std::get<0>(terms) + "->" + std::get<1>(terms)[0];
  auto const& [vjp_inns, vjp_out_rank] = einsummable_t::parse_str(vjp_string);
  einsummable_t vjp(
    reduction_sum.inn_shapes()[0],
    vjp_inns,
    vjp_out_rank,
    scalarop_t::make_identity()
  );

  std::cout << castable_t::min << std::endl;
  std::cout << reduction_sum << std::endl;
  std::cout << vjp << std::endl;

  einsummable_t reduction_max(
    join_shape.value(),
    inns,
    out_rank,
    scalarop_t::make_identity(),
    castable_t::max
  );

  std::cout << reduction_max.inn_shapes()[0] << std::endl;
  std::cout << reduction_max << std::endl;

  string broadcast_string = vjp_string;
  std::cout << broadcast_string << std::endl;
  auto const& [bc_inns, bc_out_rank] = einsummable_t::parse_str(broadcast_string);

  einsummable_t broadcast_mm(
    reduction_max.inn_shapes()[0],
    bc_inns,
    bc_out_rank,
    scalarop_t::make_identity()
  );

  std::cout << broadcast_mm << std::endl;

  // Max min reduction 

  auto const jacobian_terms = einsummable_t::make_str_terms(bc_inns, bc_out_rank);
  string jacobian_part  = std::get<1>(terms)[0];
  string jacobian_string = reduction_max.create_reduction_vjp_string();
  auto const& [jacobi_inns, jacobi_out_rank] = einsummable_t::parse_str(jacobian_string);
  auto const& join_shp = einsummable_t::construct_join_shape(jacobi_inns, {reduction_max.out_shape(), reduction_max.inn_shapes()[0]});

  auto const& mask = scalarop_t::make_mask(compare_t::eq);

  std::cout << jacobian_string << std::endl;
  std::cout << jacobi_inns << " " << jacobi_out_rank << std::endl;
  std::cout << mask << std::endl;

  einsummable_t jacobian_mm(
    join_shp.value(),
    jacobi_inns,
    jacobi_out_rank,
    mask
  );

  std::cout << jacobian_mm << std::endl;

  // Mul reduction 

  einsummable_t reduction_mul(
    join_shape.value(),
    inns,
    out_rank,
    scalarop_t::make_identity(),
    castable_t::mul
  );

  auto const& div = scalarop_t::make_div();

  einsummable_t jacobian_mul(
    reduction_mul.inn_shapes()[0],
    jacobi_inns,
    jacobi_out_rank,
    div
  );

  std::cout << jacobian_mul << std::endl;

  string vjpstr = "i,ij->ij";
  auto const& [vinns, vor] = einsummable_t::parse_str(vjpstr);
  vector<uint64_t> innshp1 = {2};
  vector<uint64_t> innshp2 = {2, 5};
  auto const& join_shapez = einsummable_t::construct_join_shape(vinns, {innshp1, innshp2});

  einsummable_t test(
    join_shapez.value(),
    vinns,
    vor,
    scalarop_t::make_mul()
  );

  std::cout << test << std::endl;
}

void print_graph() {

  vector<uint64_t> input_shape1 = {4, 5, 2, 3};
  vector<uint64_t> input_shape2 = {4, 5, 3, 5};
  vector<uint64_t> input_shape3 = {4, 5, 2, 5};

  graph_constructor_t graph;
  dtype_t dtype = default_dtype();

  int i1 = graph.insert_input(input_shape1
  );

  int i2 = graph.insert_input(input_shape2
  );

  string e1 = "deab,debc->deac";
  auto const& [inns, out_rank] = einsummable_t::parse_str(e1);
  auto const& join_shape = einsummable_t::construct_join_shape(inns, {input_shape1, input_shape2});

  einsummable_t bmm(
    join_shape.value(),
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  );

  int i3 = graph.insert_einsummable(bmm, {i1, i2});

  string e2 = "abcd->abcd";
  auto const& [inns1, out_rank1] = einsummable_t::parse_str(e2);
  auto const& join_shape1 = einsummable_t::construct_join_shape(inns1, {bmm.out_shape()});

  //int i4 = graph.insert_formation(i3, true);

  einsummable_t relu(
    join_shape1.value(),
    inns1,
    out_rank1,
    scalarop_t::make_relu()
  );

  int i5 = graph.insert_einsummable(
    relu,
     {i3});

  auto const& a = graph.graph.backprop(i5, {i1});
  
  taskgraph_t taskgraph;

  auto pls = graph.get_placements();
  for(int i = 0; i != pls.size(); ++i) {
    DOUT(i << " " << pls[i].partition);
  }

  // just set every block in every location to something
  // random
  set_seed(0);
  for(auto& placement: pls) {
    for(auto& loc: placement.locations.get()) {
      loc = runif(1);
    }
  }

  auto [_0, _1, _taskgraph] = taskgraph_t::make(graph.graph, pls);
  taskgraph = _taskgraph;

  auto [_2, _3, memgraph] = memgraph_t::make(taskgraph, {0}, {2000});
  // ^ note that memsizes and allocat_settings not being provided

  {
    std::ofstream f("gprint.gv");
    graph.graph.print_graphviz(f);
    DOUT("wrote g.gv");
  }

  {
    std::ofstream f("tgprint.gv");
    taskgraph.print_graphviz(f);
    DOUT("wrote tg.gv");
  }

  {
    std::ofstream f("mgprint.gv");
    memgraph.print_graphviz(f);
    DOUT("printed mg.gv");
  }
}

void bbbbb() {
  vector<uint64_t> input_shape1 = {4, 5, 2, 3};
  vector<uint64_t> input_shape2 = {4, 5, 3, 5};
  vector<uint64_t> input_shape3 = {4, 5, 2, 5};

  graph_t graph;
  dtype_t dtype = default_dtype();

  int i1 = graph.insert_input(input_shape1
  );

  int i2 = graph.insert_input(input_shape2
  );

  string e1 = "deab,debc->deac";
  auto const& [inns, out_rank] = einsummable_t::parse_str(e1);
  auto const& join_shape = einsummable_t::construct_join_shape(inns, {input_shape1, input_shape2});

  einsummable_t bmm(
    join_shape.value(),
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  );

  int i3 = graph.insert_einsummable(bmm, {i1, i2});

  string e2 = "abcd->abcd";
  auto const& [inns1, out_rank1] = einsummable_t::parse_str(e2);
  auto const& join_shape1 = einsummable_t::construct_join_shape(inns1, {bmm.out_shape()});

  //int i4 = graph.insert_formation(i3, true);

  einsummable_t relu(
    join_shape1.value(),
    inns1,
    out_rank1,
    scalarop_t::make_relu()
  );

  int i5 = graph.insert_einsummable(
    relu,
     {i3});

  vector<int> grads = graph.backprop(i5, {i1});

  string ds = write_with_ss(dtype);

  scalar_t lr = scalar_t(0.05).convert(dtype);

  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole|"+ds+"@0"),
      scalarop_t::make_scale(lr)
    }
  );

  auto const& ws = graph.nodes[i1];

  einsummable_t grad_update(
    ws.op.out_shape(),
    {{0,1,2,3}, {0,1,2,3}},
    4,
    gradupdate
  );

  int const& g = grads[0];
  auto const& node = graph.nodes[g];
  graph.insert_einsummable(grad_update, {i1, g});  

  {
    std::ofstream f("gprint.gv");
    graph.print_graphviz(f);
    DOUT("wrote g.gv");
  }
}

void example_memgraph() {

  vector<uint64_t> input_shape1 = {4, 5, 2, 3};
  vector<uint64_t> input_shape2 = {4, 5, 3, 5};

  string e1 = "deab,debc->deac";
  auto const& [inns, out_rank] = einsummable_t::parse_str(e1);
  auto const& join_shape = einsummable_t::construct_join_shape(inns, {input_shape1, input_shape2});

  einsummable_t bmm(
    join_shape.value(),
    inns,
    out_rank,
    scalarop_t::make_mul(),
    castable_t::add
  );

  memgraph_t memgraph(1, 1, {0});

  touchdim_t tdim1 = {0, 0, 600, 0, 600};
  touchdim_t tdim2 = {0, 0, 600, 360, 240};

  memgraph_t::inputmem_t inputmem1 = {0, 0, 360};
  memgraph_t::inputmem_t inputmem2 = {0, 360, 240};
  memgraph_t::alloc_t alloc1 = {0, 600, 600};
  touch_t touch1 = {{tdim1}, castable_t::add};
  touch_t touch2 = {{tdim2}, castable_t::add};
  memgraph_t::apply_t apply1 = {0, {alloc1.as_mem(), inputmem1.as_mem()}, touch1, 0};
  memgraph_t::apply_t apply2 = {0, {alloc1.as_mem(), inputmem2.as_mem()}, touch2, 0};
  memgraph_t::del_t del1 = {0, 0, 360};
  memgraph_t::del_t del2 = {0, 360, 240};
  memgraph_t::partialize_t partialize1 = {0, 600, 600};
  memgraph_t::alloc_t alloc2 = {0, 0, 600};
  memgraph_t::inputsto_t inputsto1 = {0, 0, 0};
  memloc_t memloc1 = {0, 1560, 80};
  memgraph_t::load_t load1 = {inputsto1.as_stoloc(), memloc1};
  memloc_t dst = {0, 600, 1200};
  memgraph_t::apply_t apply3 = {0, {memloc1.as_mem(), alloc2.as_mem(), dst.as_mem()}, bmm, 0};
  stoloc_t stoloc1 = {0, 1};
  memgraph_t::evict_t evict1 = {dst, stoloc1};

  int m0 = memgraph.insert(inputmem1, {});
  int m1 = memgraph.insert(inputmem2, {});
  int m2 = memgraph.insert(alloc1, {});
  int m3 = memgraph.insert(apply1, {m0, m2});
  int m4 = memgraph.insert(apply2, {m1, m2});
  int m5 = memgraph.insert(del1, {m3});
  int m6 = memgraph.insert(del2, {m4});
  int m7 = memgraph.insert(partialize1, {m3, m4});
  int m8 = memgraph.insert(alloc2, {m5, m6});
  int m9 = memgraph.insert(inputsto1, {});
  int m10 = memgraph.insert(load1, {m9, m8});
  int m11 = memgraph.insert(apply3, {m10, m7});
  int m12 = memgraph.insert(evict1, {m11});

  {
    std::ofstream f("mghand.gv");
    memgraph.print_graphviz(f);
    DOUT("printed mg.gv");
  }
}


int main() {
  //mul_derivative();
  //print_graph();
  // bbbbb();
  example_memgraph();
  // contraction_test();
  //eduction_test();
  //test_mm();
}