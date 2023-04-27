#include "scalarop.h"

void istream_expect(std::istream& inn, string const& xs) {
  for(auto const& x: xs) {
    if(x != inn.get()) {
      throw std::runtime_error("expected " + xs);
    }
  }
}

bool compare(compare_t c, float lhs, float rhs) {
  if(c == compare_t::lt) {
    return lhs < rhs;
  } else if(c == compare_t::gt) {
    return lhs > rhs;
  } else if(c == compare_t::eq) {
    return lhs == rhs;
  } else if(c == compare_t::le) {
    return lhs <= rhs;
  } else if(c == compare_t::ge) {
    return lhs >= rhs;
  } else {
    throw std::runtime_error("should not reach");
  }
}

namespace scalar_ns {

bool op_t::is_constant() const { return std::holds_alternative< constant >(op); }
bool op_t::is_hole()     const { return std::holds_alternative< hole     >(op); }
bool op_t::is_add()      const { return std::holds_alternative< add      >(op); }
bool op_t::is_mul()      const { return std::holds_alternative< mul      >(op); }
bool op_t::is_exp()      const { return std::holds_alternative< exp      >(op); }
bool op_t::is_power()    const { return std::holds_alternative< power    >(op); }
bool op_t::is_sqrt()     const { return std::holds_alternative< sqrt     >(op); }
bool op_t::is_ite()      const { return std::holds_alternative< ite      >(op); }

float op_t::get_constant() const { return std::get<constant>(op).value; }

int op_t::get_which_input() const { return std::get<hole>(op).arg; }

int op_t::get_power() const { return std::get<power>(op).to_the; }

compare_t op_t::get_ite_compare() const { return std::get<ite>(op).compare; }

int op_t::num_inputs() const {
  if(is_constant() || is_hole()) {
    return 0;
  }
  if(is_power() || is_sqrt() || is_exp()) {
    return 1;
  }
  if(is_add() || is_mul()) {
    return 2;
  }
  if(is_ite()) {
    return 4; // if (compare v0 v1) then v2 else v3
  }
  throw std::runtime_error("should not reach: num_inputs");
}

float op_t::eval(vector<float> const& xs) const {
  if(xs.size() != num_inputs()) {
    throw std::runtime_error("invalid op_t::eval");
  }
  if(is_constant()) {
    return get_constant();
  }
  if(is_hole()) {
    throw std::runtime_error("cannot eval inputs; no variable state here");
  }
  if(is_add()) {
    return xs[0] + xs[1];
  }
  if(is_mul()) {
    return xs[0] * xs[1];
  }
  if(is_exp()) {
    return std::exp(xs[0]);
  }
  if(is_power()) {
    return std::pow(xs[0], get_power());
  }
  if(is_sqrt()) {
    return std::sqrt(xs[0]);
  }
  if(is_ite()) {
    if(compare(get_ite_compare(), xs[0], xs[1])) {
      return xs[2];
    } else {
      return xs[3];
    }
  }
  throw std::runtime_error("should not reach");
}

float node_t::eval(vector<float> const& inputs) const {
  if(op.is_hole()) {
    return inputs[op.get_which_input()];
  }

  vector<float> cs;
  cs.reserve(children.size());
  for(auto const& child: children) {
    cs.push_back(child.eval(inputs));
  }

  return op.eval(cs);
}

node_t node_t::derivative(int arg) const {
  if(op.is_constant()) {
    return node_t {
      .op = parse_with_ss<op_t>("constant{0}"),
      .children = {}
    };
  }

  if(op.is_hole()) {
    if(arg == op.get_which_input()) {
      return node_t {
        .op = parse_with_ss<op_t>("constant{1}"),
        .children = {}
      };
    } else {
      return node_t {
        .op = parse_with_ss<op_t>("constant{0}"),
        .children = {}
      };
    }
  }

  if(op.is_add()) {
    node_t const& lhs = children[0];
    node_t const& rhs = children[1];

    node_t deri_lhs = lhs.derivative(arg);
    node_t deri_rhs = rhs.derivative(arg);

    return node_t {
      .op = parse_with_ss<op_t>("+"),
      .children = {deri_lhs, deri_rhs}
    };
  }

  if(op.is_mul()) {
    node_t const& lhs = children[0];
    node_t const& rhs = children[1];

    node_t deri_lhs = lhs.derivative(arg);
    node_t deri_rhs = rhs.derivative(arg);

    string s_lhs      = write_with_ss(lhs);
    string s_rhs      = write_with_ss(rhs);

    string s_deri_lhs = write_with_ss(deri_lhs);
    string s_deri_rhs = write_with_ss(deri_rhs);

    // Gradient of d(L(x)R(x))/dx = L' R  + L R'

    string term_lhs = "*[" + s_deri_lhs + "," + s_rhs      + "]";
    string term_rhs = "*[" + s_lhs      + "," + s_deri_rhs + "]";

    return parse_with_ss<node_t>("+[" + term_lhs + "," + term_rhs + "]");
  }

  if(op.is_exp()) {
    // e^{f(x)} => e^{f(x)} * f'(x)
    node_t const& inn = children[0];

    node_t deri_inn = inn.derivative(arg);

    string s_inn      = write_with_ss(inn);

    string s_deri_inn = write_with_ss(deri_inn);

    return parse_with_ss<node_t>("*[" + s_inn + "," + s_deri_inn + "]");
  }

  if(op.is_power()) {
    // I(x)^i => i * { (I(x) ^{i-1}) * I'(x) }
    //           A     B               C

    int i = op.get_power();

    if(i == 0) {
      return parse_with_ss<node_t>("constant{0}");
    }

    node_t const& inn      = children[0];
    node_t deri_inn = inn.derivative(arg);

    if(i == 1) {
      return deri_inn;
    }

    string s_inn      = write_with_ss(inn);

    string A = "constant{" + write_with_ss(i) + "}";
    string B = "power{" + write_with_ss(i-1) + "}[" + s_inn + "]";
    string C = write_with_ss(deri_inn);

    string BC = "*[" + B + "," + C + "]";
    string ABC = "*[" + A + "," + BC + "]";

    return parse_with_ss<node_t>(ABC);
  }

  if(op.is_sqrt()) {
    throw std::runtime_error("not implemented");
  }

  if(op.is_ite()) {
    // compare(x0, x1) ? x2  : x3  has a derivative of
    // compare(x0, x1) ? x2' : x3'
    string s0 = write_with_ss(children[0]);
    string s1 = write_with_ss(children[1]);
    string deri_s2 = write_with_ss(children[2].derivative(arg));
    string deri_s3 = write_with_ss(children[3].derivative(arg));

    string compare = write_with_ss(op.get_ite_compare());

    string ret = "ite_" + compare +
      "[" + s0 + "," + s1 + "," + deri_s2 + "," + deri_s3 + "]";

    return parse_with_ss<node_t>(ret);
  }

  throw std::runtime_error("should not reach");
}

node_t node_t::simplify() const {
  node_t r0 = simplify_once();
  if(r0 == *this) {
    return r0;
  }

  node_t r1 = r0.simplify_once();
  while(true) {
    if(r0 == r1) {
      return r1;
    }
    r0 = r1.simplify_once();
    std::swap(r0, r1);
  }
}

node_t node_t::simplify_once() const {
  if(op.is_hole() || op.is_constant()) {
    return *this;
  }

  // Case: Has no inputs (and therefore should be a constant)
  {
    set<int> holes; which_inputs(holes);
    if(holes.size() == 0) {
      float val = eval({});
      string constant = ("constant{" + write_with_ss(val) + "}");
      return parse_with_ss<node_t>(constant);
    }
  }

  vector<node_t> new_children =
    vector_from_each_method(children, node_t, simplify);

  // Case: Add
  if(op.is_add()) {
    // Check for 0 + x or x + 0
    node_t& lhs = new_children[0];
    node_t& rhs = new_children[1];
    if(lhs.op.is_constant() && lhs.op.get_constant() == 0.0) {
      return rhs;
    }
    if(rhs.op.is_constant() && rhs.op.get_constant() == 0.0) {
      return lhs;
    }
  }

  // TODO: also cover x^i * x^j
  //       and        x   * x
  //       and        x   * x^i
  //       and        x^i * x

  // Case: Mul
  if(op.is_mul()) {
    // Check for 0*x or x*0
    node_t& lhs = new_children[0];
    node_t& rhs = new_children[1];
    if(
      (lhs.op.is_constant() && lhs.op.get_constant() == 0.0) ||
      (rhs.op.is_constant() && rhs.op.get_constant() == 0.0))
    {
      return parse_with_ss<node_t>("constant{0}");
    }

    // Check for 1*x or x*1
    if(lhs.op.is_constant() && lhs.op.get_constant() == 1.0) {
      return rhs;
    }
    if(rhs.op.is_constant() && rhs.op.get_constant() == 1.0) {
      return lhs;
    }
  }

  // Case: Exp
  // e^0 is already covered since there'd be no inputs

  // Case: Power
  if(op.is_power()) {
    // check for x^0 or x^1
    int i = op.get_power();
    if(i == 0) {
      return parse_with_ss<node_t>("constant{1}");
    }
    if(i == 1) {
      node_t& inn = new_children[0];
      return inn;
    }
  }

  // Case: Sqrt
  // TODO: change power to be a float

  // Case: Ite
  if(op.is_ite()) {
    // check for ite z z x y
    // and       ite x y z z
    node_t& s0 = new_children[0];
    node_t& s1 = new_children[1];
    node_t& s2 = new_children[2];
    node_t& s3 = new_children[3];
    if(s0 == s1) {
      compare_t c = op.get_ite_compare();
      if(c == compare_t::eq) {
        return s2;
      }
    }
    if(s2 == s3) {
      return s2;
    }
  }

  return node_t {
    .op = op,
    .children = new_children
  };
}

void node_t::which_inputs(set<int>& items) const {
  if(op.is_hole()) {
    items.insert(op.get_which_input());
  }
  for(auto const& child: children) {
    child.which_inputs(items);
  }
}

int node_t::max_hole() const {
  if(op.is_hole()) {
    return op.get_which_input();
  }

  int ret = -1;
  for(auto const& child: children) {
    ret = std::max(ret, child.max_hole());
  }
  return ret;
}

int node_t::num_inputs() const {
  return max_hole() + 1;
}

void node_t::increment_holes(int incr) {
  if(op.is_hole()) {
    int& arg = std::get<op_t::hole>(op.op).arg;
    arg += incr;
  } else {
    for(auto& child: children) {
      child.increment_holes(incr);
    }
  }
}

void node_t::remap_holes(map<int, int> const& fmap) {
  if(op.is_hole()) {
    int& arg = std::get<op_t::hole>(op.op).arg;
    arg += fmap.at(arg);
  } else {
    for(auto& child: children) {
      child.remap_holes(fmap);
    }
  }
}

void node_t::replace_at_holes(vector<node_t> const& replace_nodes)
{
  if(op.is_hole()) {
    int arg = std::get<op_t::hole>(op.op).arg;
    *this = replace_nodes[arg];
  } else {
    for(auto& child: children) {
      child.replace_at_holes(replace_nodes);
    }
  }
}

} // scalar_ns

scalarop_t::scalarop_t() {}

scalarop_t::scalarop_t(scalar_ns::node_t const& node)
  : node(node.simplify())
{}

float scalarop_t::eval(vector<float> const& inputs) const {
  return node.eval(inputs);
}

scalarop_t scalarop_t::derivative(int arg) const {
  return scalarop_t(node.derivative(arg));
}

scalarop_t scalarop_t::simplify() const {
  return scalarop_t(node.simplify());
}

void scalarop_t::remap_inputs(map<int, int> const& remap) {
  node.remap_holes(remap);
}

set<int> scalarop_t::which_inputs() const {
  set<int> ret;
  node.which_inputs(ret);
  return ret;
}

int scalarop_t::num_inputs() const {
  return node.num_inputs();
}

bool scalarop_t::is_constant() const {
  return num_inputs() == 0;
}


bool scalarop_t::is_unary() const {
  return num_inputs() == 1;
}

bool scalarop_t::is_binary() const {
  return num_inputs() == 2;
}

bool scalarop_t::is_castable() const {
  string self = write_with_ss(*this);

  vector<string> xs {
    "*[hole@0,hole@1]",
    "*[hole@1,hole@0]",
    "+[hole@0,hole@1]",
    "+[hole@1,hole@0]",
    // A lot of ways to write min and max...
    "ite_<[hole@0,hole@1,hole@0,hole@1]",
    "ite_<=[hole@0,hole@1,hole@0,hole@1]",
    "ite_>[hole@0,hole@1,hole@0,hole@1]",
    "ite_>=[hole@0,hole@1,hole@0,hole@1]",
    "ite_<[hole@0,hole@1,hole@1,hole@0]",
    "ite_<=[hole@0,hole@1,hole@1,hole@0]",
    "ite_>[hole@0,hole@1,hole@1,hole@0]",
    "ite_>=[hole@0,hole@1,hole@1,hole@0]",
    "ite_<[hole@1,hole@0,hole@0,hole@1]",
    "ite_<=[hole@1,hole@0,hole@0,hole@1]",
    "ite_>[hole@1,hole@0,hole@0,hole@1]",
    "ite_>=[hole@1,hole@0,hole@0,hole@1]",
    "ite_<[hole@1,hole@0,hole@1,hole@0]",
    "ite_<=[hole@1,hole@0,hole@1,hole@0]",
    "ite_>[hole@1,hole@0,hole@1,hole@0]",
    "ite_>=[hole@1,hole@0,hole@1,hole@0]"
  };

  for(auto const& x: xs) {
    if(self == x) {
      return true;
    }
  }

  return false;
}

bool scalarop_t::is_constant_of(float val) const {
  return node.op.is_constant() && node.op.get_constant() == val;
}

// Example: combining_op = (y0 * y1) + y2, ops = (x0 + x1, x0 + x1, 7*x0), this replaces
// y0 with x0 + x1 and
// y1 with x2 + x3 and
// y2 with 7*x4
// to get
//   ((x0 + x1) * (x2 + x3)) + (7*x4)
// Note that each input op ends up having distinct inputs
scalarop_t scalarop_t::combine(scalarop_t combining_op, vector<scalarop_t> const& inn_ops) {
  if(combining_op.num_inputs() != inn_ops.size()) {
    throw std::runtime_error("cannot combine");
  }

  if(combining_op.num_inputs() == 0) {
    return combining_op;
  }

  vector<node_t> inn_nodes = vector_from_each_member(inn_ops, node_t, node);

  // TODO TODO give node_t a number of inputs and remove num_inputs code from
  //           scalarop_t

  int n = inn_nodes.size();
  if(n > 1) {
    int offset = inn_nodes[0].num_inputs();
    for(int i = 1; i != n; ++i) {
      auto& inn_node = inn_nodes[i];
      int num_here = inn_node.num_inputs();
      inn_node.increment_holes(offset);
      offset += num_here;
    }
  }

  combining_op.node.replace_at_holes(inn_nodes);
  return combining_op;
}

scalarop_t scalarop_t::from_string(string const& str) {
  return parse_with_ss<scalarop_t>(str);
}

// x0 + x1
scalarop_t scalarop_t::make_add() {
  return parse_with_ss<scalarop_t>("+[hole@0,hole@1]");
}
// x0 * x1
scalarop_t scalarop_t::make_mul() {
  return parse_with_ss<scalarop_t>("*[hole@0,hole@1]");
}
// min(x0, x1)
scalarop_t scalarop_t::make_min() {
  return parse_with_ss<scalarop_t>("ite_<[hole@0,hole@1,hole@0,hole@1]");
}
// max(x0, x1)
scalarop_t scalarop_t::make_max() {
  return parse_with_ss<scalarop_t>("ite_>[hole@0,hole@1,hole@0,hole@1]");
}
// xn * val
scalarop_t scalarop_t::make_scale_which(float val, int arg) {
  string hole = "hole@" + write_with_ss(arg);
  string constant = "constant{" + write_with_ss(val) + "}";
  return parse_with_ss<scalarop_t>("*[" + hole + "," + constant + "]");
}
// x0 * val
scalarop_t scalarop_t::make_scale(float val) {
  return make_scale_which(val, 0);
}
// x0 - x1
scalarop_t scalarop_t::make_sub() {
  string negate = write_with_ss(make_scale_which(-1.0, 1));
  string op = "+[hole@0," + negate + "]";
  return parse_with_ss<scalarop_t>(op);
}
// x0 + val
scalarop_t scalarop_t::make_increment(float val) {
  string constant = "constant{" + write_with_ss(val) + "}";
  return parse_with_ss<scalarop_t>("+[hole@0," + constant + "]");
}

scalarop_t scalarop_t::make_relu() {
  string arg0 = "hole@0";
  string zero = "constant{0}";
  string ite = "ite_<[" + arg0 + "," + zero + "," + zero + "," + arg0 + "]";
  return parse_with_ss<scalarop_t>(ite);
}

scalarop_t scalarop_t::make_relu_deriv() {
  return make_relu().derivative(0);
}

scalarop_t scalarop_t::make_from_castable(castable_t c) {
  if(c == castable_t::add) {
    return make_add();
  }  else if(c == castable_t::mul) {
    return make_mul();
  } else if(c == castable_t::min) {
    return make_min();
  } else if(c == castable_t::max) {
    return make_max();
  } else {
    throw std::runtime_error("should not reach");
  }
}

bool operator==(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs) {
  return write_with_ss(lhs) == write_with_ss(rhs);
}

bool operator!=(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs) {
  return !(lhs == rhs);
}

bool operator==(scalarop_t const& lhs, scalarop_t const& rhs) {
  return write_with_ss(lhs) == write_with_ss(rhs);
}

bool operator!=(scalarop_t const& lhs, scalarop_t const& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, compare_t const& c) {
  if(c == compare_t::lt) {
    out << "<";
  } else if(c == compare_t::gt) {
    out << ">";
  } else if(c == compare_t::eq) {
    out << "==";
  } else if(c == compare_t::le) {
    out << "<=";
  } else if(c == compare_t::ge) {
    out << ">=";
  } else {
    throw std::runtime_error("should not reach");
  }

  return out;
}

std::istream& operator>>(std::istream& inn, compare_t& compare) {
  char c0 = inn.get();
  char c1 = inn.peek();

  if(c0 == '<') {
    if(c1 == '=') {
      inn.get();
      compare = compare_t::le;
    } else {
      compare = compare_t::lt;
    }
  } else if(c0 == '>') {
    if(c1 == '=') {
      inn.get();
      compare = compare_t::ge;
    } else {
      compare = compare_t::gt;
    }
  } else if(c0 == '=') {
    if(c1 == '=') {
      inn.get();
      compare = compare_t::eq;
    } else {
      throw std::runtime_error("invalid parse: compare_t");
    }
  } else {
    throw std::runtime_error("should not reach: parsing compare_t");
  }

  return inn;
}

std::ostream& operator<<(std::ostream& out, scalar_ns::op_t const& op) {
  if(op.is_constant()) {
    out << "constant{" << op.get_constant() << "}";
  } else if(op.is_hole()) {
    out << "hole@" << op.get_which_input();
  } else if(op.is_add()) {
    out << "+";
  } else if(op.is_mul()) {
    out << "*";
  } else if(op.is_exp()) {
    out << "exp";
  } else if(op.is_power()) {
    out << "power{" << op.get_power() << "}";
  } else if(op.is_sqrt()) {
    out << "sqrt";
  } else if(op.is_ite()) {
    out << "ite_" << op.get_ite_compare();
  } else {
    throw std::runtime_error("should not reach");
  }

  return out;
}

std::istream& operator>>(std::istream& inn, scalar_ns::op_t& op) {
  char c = inn.peek();
  if(c == 'c') {
    istream_expect(inn, "constant{");
    float v;
    inn >> v;
    istream_expect(inn, "}");
    op.op = scalar_ns::op_t::constant{ .value = v };
  } else if(c == 'h') {
    istream_expect(inn, "hole@");
    int i;
    inn >> i;
    op.op = scalar_ns::op_t::hole { .arg = i };
  } else if(c == '+') {
    inn.get();
    op.op = scalar_ns::op_t::add{ };
  } else if(c == '*') {
    inn.get();
    op.op = scalar_ns::op_t::mul{ };
  } else if(c == 'e') {
    istream_expect(inn, "exp");
    op.op = scalar_ns::op_t::exp{ };
  } else if(c == 'p') {
    istream_expect(inn, "power{");
    int i;
    inn >> i;
    op.op = scalar_ns::op_t::power{ .to_the = i };
    istream_expect(inn, "}");
  } else if(c == 's') {
    istream_expect(inn, "sqrt");
    op.op = scalar_ns::op_t::sqrt { };
  } else if(c == 'i') {
    istream_expect(inn, "ite_");
    compare_t c;
    inn >> c;
    op.op = scalar_ns::op_t::ite{ .compare = c };
  } else {
    throw std::runtime_error("should not happen");
  }
  return inn;
}

std::ostream& operator<<(std::ostream& out, scalar_ns::node_t const& node) {
  out << node.op;
  if(node.children.size() == 0) {
    return out;
  }
  out << "[";
  out << node.children[0];
  if(node.children.size() > 1) {
    for(int i = 1; i != node.children.size(); ++i) {
      out << "," << node.children[i];
    }
  }
  out << "]";

  return out;
}

std::istream& operator>>(std::istream& inn, scalar_ns::node_t& node) {
  node.children.resize(0);

  inn >> node.op;

  int n = node.op.num_inputs();
  if(n == 0) {
    return inn;
  }

  istream_expect(inn, "[");
  {
    node.children.emplace_back();
    inn >> node.children.back();
  }
  if(n > 1) {
    for(int i = 1; i != n; ++i) {
      istream_expect(inn, ",");
      node.children.emplace_back();
      inn >> node.children.back();
    }
  }
  istream_expect(inn, "]");
  return inn;
}

std::ostream& operator<<(std::ostream& out, scalarop_t const& op) {
  out << op.node;
  return out;
}

std::istream& operator>>(std::istream& inn, scalarop_t& op) {
  scalar_ns::node_t node;
  inn >> node;
  op = scalarop_t(node);
  return inn;
}

