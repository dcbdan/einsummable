# include "../../src/einsummable/simplescalarop.h"

// // define some scalarops, convert them
scalarop_t make_scalarop(){
  // x1 + -1 * x2
  scalarop_t one = scalarop_t::make_constant(scalar_t::one(dtype_t::f32));
  scalarop_t neg_one = scalarop_t::make_constant(scalar_t::negative_one(dtype_t::f32));
  scalarop_t div = scalarop_t::make_div(dtype_t::f32);
  scalarop_t exp = scalarop_t::make_exp(dtype_t::f32);
  scalarop_t mul = scalarop_t::make_mul(dtype_t::f32);
  scalarop_t sub = scalarop_t::make_sub(dtype_t::f32);
  scalarop_t add = scalarop_t::make_add(dtype_t::f32);
  scalarop_t log = scalarop_t::make_log(dtype_t::f32);
  scalarop_t arg = scalarop_t::make_arg(0, dtype_t::f32);
  scalarop_t arg1 = scalarop_t::make_arg(1, dtype_t::f32);

  scalarop_t ret = scalarop_t::combine(mul, {neg_one, arg});
  ret = scalarop_t::combine(add, {arg, ret});
  return ret;
}


void test(){
  DOUT("---Converting---");
  scalarop_t op = make_scalarop();
  optional<list_simple_scalarop_t> sop = list_simple_scalarop_t::make(op);
  if (sop){
    DOUT("Number of scalarops: " << sop.value().ops.size());
    DOUT("---Converting back---");
    auto new_op = sop.value().to_scalarop();
    std::cout << "Original op: " << op << std::endl;
    std::cout << "translated op: " << new_op << std::endl;
    if (op != new_op){
      DOUT("Conversion failed");
    }
    else {
      DOUT("Conversion succeeded");
    }
  } else {
    throw std::runtime_error("Conversion failed");
  }
}

int main(){
  test();
  return 0;
}