#include "../src/base/setup.h"
#include "../src/einsummable/scalarop.h"

scalar_t fl(float v) { return scalar_t(v); }

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
    scalarop_t op = scalarop_t::make_scale(fl(3.5));
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
    scalarop_t op = scalarop_t::make_increment(fl(9.3));
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
  }
  {
    std::cout << "FF2 (x0 + 0) * (x1 * 0)" << std::endl;
    std::string s = "*[+[hole|f32@0,constant{f32|0}],*[hole|f32@1,constant{f32|0}]]";
    scalarop_t op = parse_with_ss<scalarop_t>(s);
    std::cout << op << std::endl;
    std::cout << op.derivative(0) << std::endl;
    std::cout << op.derivative(1) << std::endl;
  }
  {
    std::cout << "FF3 (x0 + 0) * (x0 * 1)" << std::endl;
    std::string s = "*[+[hole|f32@0,constant{f32|0}],*[hole|f32@0,constant{f32|1}]]";
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
    scalarop_t scale = scalarop_t::make_scale(fl(7.0));
    scalarop_t top   = scalarop_t::from_string("+[*[hole|f32@0,hole|f32@1],hole|f32@2]");
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
  {
    std::cout << "CONJUGATE" << std::endl;
    scalarop_t op = scalarop_t::make_conjugate();
    std::cout << op << std::endl;
    std::cout << parse_with_ss<scalarop_t>(write_with_ss(op)) << std::endl;
    scalarop_t identity = scalarop_t::combine(op, {op});
    std::cout << "identity: " << identity << std::endl;
  }
  {
    std::cout << "PROJECT REAL" << std::endl;
    scalarop_t op = scalarop_t::make_project_real();
    std::cout << op << std::endl;
    std::cout << parse_with_ss<scalarop_t>(write_with_ss(op)) << std::endl;
  }
  {
    std::cout << "PROJECT IMAG" << std::endl;
    scalarop_t op = scalarop_t::make_project_imag();
    std::cout << op << std::endl;
    std::cout << parse_with_ss<scalarop_t>(write_with_ss(op)) << std::endl;
  }
  {
    std::cout << "CPLEX" << std::endl;
    scalarop_t op = scalarop_t::make_complex();
    std::cout << op << std::endl;
    std::cout << parse_with_ss<scalarop_t>(write_with_ss(op)) << std::endl;
  }
  {
    std::cout << "... wirtinger derivs for x*y" << std::endl;
    scalarop_t op = scalarop_t::make_mul(dtype_t::c64);
    std::cout << op.wirtinger_derivative(0, true) << std::endl;
    std::cout << op.wirtinger_derivative(0, false) << std::endl;
    std::cout << op.wirtinger_derivative(1, true) << std::endl;
    std::cout << op.wirtinger_derivative(1, false) << std::endl;
  }
  {
    std::cout << "VARIABLES" << std::endl;
    // f(x,y) = x - y*learning_rate
    scalarop_t op = scalarop_t::combine(
      scalarop_t::make_sub(),
      {
        scalarop_t::make_identity(),
        scalarop_t::make_scale("learning_rate")
      }
    );
    std::cout << op << std::endl;
    std::cout << parse_with_ss<scalarop_t>(write_with_ss(op)) << std::endl;

    map<string, scalar_t> vars;
    vars.insert({"learning_rate", scalar_t(float(1e-3))});
    vector<scalar_t> args {
      scalar_t(float(1.0)),
      scalar_t(float(2.0))
    };
    std::cout << op.eval(args, vars) << std::endl;
  }
}

void main02() {
  float16_t a(3.4), b(5);
  float16_t c = a * b;
  c += 3;
  if(c > a) {
    std::cout << c << std::endl;
  }
  DOUT("e^3.4 half float " << half_float::exp(a));
  DOUT("e^3.4 float      " << std::exp(float(3.4)));
}

int main() {
  main01();
}
