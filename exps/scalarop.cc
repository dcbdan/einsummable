#include "../src/einsummable/scalarop.h"

int main() {
  using namespace scalar_ns;

  {
    std::cout << "ADD" << std::endl;
    scalarop_t op = scalarop_t::make_add();
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
    std::cout << op.gradient(1) << std::endl;
  }
  {
    std::cout << "MUL" << std::endl;
    scalarop_t op = scalarop_t::make_mul();
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
    std::cout << op.gradient(1) << std::endl;
  }
  {
    std::cout << "x -> x*3.5" << std::endl;
    scalarop_t op = scalarop_t::make_scale(3.5);
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
    std::cout << op.gradient(1) << std::endl;
  }
  {
    std::cout << "SUB" << std::endl;
    scalarop_t op = scalarop_t::make_sub();
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
    std::cout << op.gradient(1) << std::endl;
  }
  {
    std::cout << "FF1 x -> x + 9.3" << std::endl;
    scalarop_t op = scalarop_t::make_increment(9.3);
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
  }
  {
    std::cout << "FF2 (x0 + 0) * (x1 * 0)" << std::endl;
    std::string s = "*[+[hole@0,constant{0}],*[hole@1,constant{0}]]";
    scalarop_t op = parse_with_ss<scalarop_t>(s);
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
    std::cout << op.gradient(1) << std::endl;
  }
  {
    std::cout << "FF3 (x0 + 0) * (x0 * 1)" << std::endl;
    std::string s = "*[+[hole@0,constant{0}],*[hole@0,constant{1}]]";
    scalarop_t op = parse_with_ss<scalarop_t>(s);
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
  }
  {
    std::cout << "FF4 (x0 + x1) * (x2 + x3)" << std::endl;
    scalarop_t add = scalarop_t::make_add();
    op_t __op = parse_with_ss<op_t>("*");
    scalarop_t op = scalarop_t::combine(__op, {add, add});
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
    std::cout << op.gradient(1) << std::endl;
    std::cout << op.gradient(2) << std::endl;
    std::cout << op.gradient(3) << std::endl;
  }
  {
    std::cout << "RELU" << std::endl;
    scalarop_t op = scalarop_t::make_relu();
    std::cout << op << std::endl;
    std::cout << op.gradient(0) << std::endl;
  }
}
