# include "../../src/einsummable/simplescalarop.h"

// // define some scalarops, convert them
scalarop_t make_scalarop(){
  // 1 - (1 / (e^-x))
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
  // 1) log(x - 1) + (1 / (e^-x))
  // -x
  scalarop_t right = scalarop_t::combine(mul, {neg_one, arg});
  // e^-x
  right = scalarop_t::combine(exp, {right});
  // 1 / (e^-x)
  right = scalarop_t::combine(div, {one, right});
  // log(x - 1) + (1 / (e^-x))
  scalarop_t left = scalarop_t::combine(sub, {arg, one});
  left = scalarop_t::combine(log, {left});
  scalarop_t ret = scalarop_t::combine(add, {left, right});
  return ret;
}


void test(){
  DOUT("---Converting---");
  scalarop_t op = make_scalarop();
  optional<list_simple_scalarop_t> sop = list_simple_scalarop_t::make(op);
  if (sop){
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