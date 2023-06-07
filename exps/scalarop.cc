#include "../src/einsummable/scalarop.h"

void main01() {
  {
    std::cout << "ADD" << std::endl;
    scalarop_t op = scalarop_t::make_add();
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
  }
  {
    std::cout << "MUL" << std::endl;
    scalarop_t op = scalarop_t::make_mul();
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
  }
  {
    std::cout << "x -> x*3.5" << std::endl;
    scalarop_t op = scalarop_t::make_scale(3.5);
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
  }
  {
    std::cout << "SUB" << std::endl;
    scalarop_t op = scalarop_t::make_sub();
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
  }
  {
    std::cout << "FF1 x -> x + 9.3" << std::endl;
    scalarop_t op = scalarop_t::make_increment(9.3);
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
  }
  {
    std::cout << "FF2 (x0 + 0) * (x1 * 0)" << std::endl;
    std::string s = "*[+[hole@0,constant{0}],*[hole@1,constant{0}]]";
    scalarop_t op = parse_with_ss<scalarop_t>(s);
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
  }
  {
    std::cout << "FF3 (x0 + 0) * (x0 * 1)" << std::endl;
    std::string s = "*[+[hole@0,constant{0}],*[hole@0,constant{1}]]";
    scalarop_t op = parse_with_ss<scalarop_t>(s);
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
  }
  {
    std::cout << "FF4 (x0 + x1) * (x2 + x3)" << std::endl;
    scalarop_t add = scalarop_t::make_add();
    scalarop_t op = scalarop_t::combine(
      scalarop_t::make_mul(),
      {add, add}
    );
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
    std::cout << op.derivative(2) << std::endl;
    std::cout << op.derivative(3) << std::endl;
  }
  {
    std::cout << "FF5 ((x0 + x1) * (x2 + x3)) + 7*x4" << std::endl;
    scalarop_t add   = scalarop_t::make_add();
    scalarop_t scale = scalarop_t::make_scale(7.0);
    scalarop_t top   = scalarop_t::from_string("+[*[hole@0,hole@1],hole@2]");
    scalarop_t op    = scalarop_t::combine(top, {add, add, scale});
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
    std::cout << op.derivative(2) << std::endl;
    std::cout << op.derivative(3) << std::endl;
    std::cout << op.derivative(4) << std::endl;
  }
  {
    std::cout << "RELU" << std::endl;
    scalarop_t op = scalarop_t::make_relu();
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
  }
}

int main() {
  float16_t a(3.4), b(5);
  float16_t c = a * b;
  c += 3;
  if(c > a) {
    std::cout << c << std::endl;
  }
}
