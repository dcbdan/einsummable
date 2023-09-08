#include "../src/einsummable/graph.h"
#include "../src/einsummable/einsummable.h"
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

  graph.backprop({b1}, {i1, i3});

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

  auto const& [mm_inns, mm_or] = einsummable_t::parse_str("ij,i->i");
  auto const& join_shape_mm = einsummable_t::construct_join_shape(mm_inns, {{4,6},{4}});
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

  auto const& stringovic = get_deri_einsum(mm, 0);

  auto const& [dinns, dor] = einsummable_t::parse_str(stringovic); 
  auto const& djs = einsummable_t::construct_join_shape(dinns, {mm.out_shape(), mm.inn_shapes()[1]});
  std::cout << mm.out_shape() << " " << mm.inn_shapes()[1] << std::endl;
  std::cout << djs.value() << std::endl;

  einsummable_t derija(
    djs.value(),
    dinns,
    dor,
    scalarop_t::make_mul(),
    castable_t::add
  );

  std::cout << "AJDEEE" << std::endl;
  std::cout << derija << std::endl;
}



int main() {
  //mul_derivative();
  contraction_test();
  //test_mm();
}