#include "simplescalarop.h"
#include "scalarop.h"
#include <iostream>
#include <tuple>

namespace scalar_ns {

bool merge_into_nodeconst_map(
  map<int, node_t const*>      & ret,
  map<int, node_t const*> const& values)
{
  for(auto const& [key, value]: values) {
    auto [iter, did_insert] = ret.insert({key, value});
    if(did_insert) {
      // Case 1: we have inserted a new key, value pair into ret
    } else {
      // Case 2: we did not insert a new key, value pair into ret.
      // It better be the case that ret[key] == value !
      node_t const& from_ret = *(iter->second);
      node_t const& from_value = *value;
      if(from_ret != from_value) {
        return false;
      }
    }
  }
  return true;
}

// Consider relu == ite compare(zero, hole0) zero hole0
// Then the recursion will hit hole0 multiple times and the
// maps will have to be checked.
//
// Consider lambda x: x + x  (hole0 + hole0)
// Here, we will have to merge {arg0, node_from_x}
optional<map<int, node_t const*>>
_pop_match(node_t const* skeleton, node_t const* node)
{
  if(skeleton->dtype != node->dtype) {
    return std::nullopt;
  }

  // This is the base case of the recursion
  if(skeleton->op.is_hole()) {
    int which_arg = skeleton->op.get_which_input();
    return map<int, node_t const*>{ { which_arg, node } };
  }

  if(skeleton->op != node->op) {
    return std::nullopt;
  }

  // now recurse!

  int num_children = skeleton->children.size();
  if(num_children != node->children.size()) {
    throw std::runtime_error(
      "should not happen: same op, different number of children");
  }
  map<int, node_t const*> ret;
  for(int which = 0; which != num_children; ++which) {
    auto maybe = _pop_match(
      &skeleton->children[which], &node->children[which]);
    if(!maybe) {
      return std::nullopt;
    }
    bool success = merge_into_nodeconst_map(ret, maybe.value());
    if(!success) {
      return std::nullopt;
    }
  }
  return ret;
}

static 
tuple<scalar_t, node_t const*> _pop_mul_scale(node_t const* node)
{
  if (node->op.is_mul()) {
    node_t const* lhs = &(node->children[0]);
    node_t const* rhs = &(node->children[1]);
    if (lhs->op.is_constant()) {
      return { lhs->op.get_constant(), rhs };
    }
    if (rhs->op.is_constant()) {
      return { rhs->op.get_constant(), lhs };
    }
  }
  return { scalar_t::one(node->dtype), node };
}

static 
optional<tuple<
  simple_scalarop_t::unary_t,
  node_t const*> >
_pop_with_unary(node_t const* node_with_scale)
{
  using unary_t = simple_scalarop_t::unary_t;
  using uop_t   = simple_scalarop_t::uop_t;

  using value_t = tuple<unary_t, node_t const*>;

  auto [scale, node] = _pop_mul_scale(node_with_scale);

  // Note: not matching with the identity uop!

  {
    vector<tuple<uop_t, scalarop_t>> ms;
    // NOTE: sort this from most complicated to least complicated
    // for example, sigmoid can be matched with rcp. exp
    ms.emplace_back(uop_t::sigmoid, scalarop_t::make_sigmoid(node->dtype));
    ms.emplace_back(uop_t::log, scalarop_t::make_log(node->dtype));
    ms.emplace_back(uop_t::exp, scalarop_t::make_exp(node->dtype));
    ms.emplace_back(uop_t::relu, scalarop_t::make_relu(node->dtype));
    ms.emplace_back(uop_t::neg, scalarop_t::make_neg(node->dtype));
    ms.emplace_back(uop_t::sqrt, scalarop_t::make_sqrt(node->dtype));
    ms.emplace_back(uop_t::rcp, scalarop_t::make_rcp(node->dtype));
    ms.emplace_back(uop_t::square, scalarop_t::make_square(node->dtype));
    

    for(auto const& [uop, m]: ms) {
      auto maybe = _pop_match(m.get_node(), node);
      if(maybe) {
        DOUT("matched: " << uop);
        auto const& val = maybe.value();
        return value_t {
          unary_t { 
            .scale = scale,
            .op = uop
          },
          val.at(0)
        };
      }
    }
  }

  // TODO: conj and what else?

  return std::nullopt;
}

static 
tuple<
  simple_scalarop_t::unary_t,
  node_t const* >
_pop_unary_success(node_t const* node) 
{
  {
    auto maybe = _pop_with_unary(node);
    if(maybe) {
      return maybe.value();
    }
  }

  return {
    simple_scalarop_t::unary_t {
      .scale = scalar_t::one(node->dtype),
      .op = simple_scalarop_t::uop_t::identity
    },
    node
  };
}

static
optional<tuple<
  simple_scalarop_t,
  vector<node_t const*> > >
_pop_with_binary_simple_scalarop(
  simple_scalarop_t::bop_t bop,
  node_t const* lhs,
  node_t const* rhs)
{
  using value_t = tuple<simple_scalarop_t, vector<node_t const*> >;

  if(lhs->op.is_constant()) {
    simple_scalarop_t::scale_t op {
      .bop = bop,
      .scale = lhs->op.get_constant()
    };
    return value_t {
      simple_scalarop_t { .op = op },
      { rhs }
    };
  }
  if(rhs->op.is_constant()) {
    return _pop_with_binary_simple_scalarop(bop, rhs, lhs);
  }

  // neither lhs or rhs is constant, but let's make extra sure
  if(lhs->num_inputs() == 0 || rhs->num_inputs() == 0) {
    throw std::runtime_error("lhs or rhs is constant; maybe simplify first!");
  }

  auto [unary_lhs, child_lhs] = _pop_unary_success(lhs);
  auto [unary_rhs, child_rhs] = _pop_unary_success(rhs);
  simple_scalarop_t::binary_t op {
    .op = bop,
    .lhs = unary_lhs,
    .rhs = unary_rhs
  };
  return value_t { 
    simple_scalarop_t { .op = op },
    { child_lhs, child_rhs }
  };
}

optional<tuple<
  simple_scalarop_t,
  vector<node_t const*> > >
pop_to_simple_scalarop(node_t const* node)
{
  using value_t = tuple<simple_scalarop_t, vector<node_t const*>>;
  using bop_t = simple_scalarop_t::bop_t;

  for(bop_t bop: { bop_t::add, bop_t::mul, bop_t::min, bop_t::max }) {
    scalarop_t m = simple_scalarop_t::bop_to_scalarop(bop, node->dtype);
    auto maybe = _pop_match(m.get_node(), node);
    if(maybe) {
      auto const& val = maybe.value();
      return _pop_with_binary_simple_scalarop(
        bop, val.at(0), val.at(1));
    }
  }

  auto maybe = _pop_with_unary(node);
  if(maybe) {
    auto const& [unary, child] = maybe.value();
    return value_t { 
      simple_scalarop_t { .op = unary },
      { child } 
    };
  }

  return std::nullopt;
}

}

scalarop_t simple_scalarop_t::unary_to_scalarop(unary_t u) {
  return scalarop_t::combine(
    scalarop_t::make_scale(u.scale),
    { uop_to_scalarop(u.op, u.scale.dtype) });
}

scalarop_t simple_scalarop_t::bop_to_scalarop(bop_t bop, dtype_t dtype) {
  if(bop == bop_t::add) {
    return scalarop_t::make_add(dtype);
  } else if(bop == bop_t::mul) {
    return scalarop_t::make_mul(dtype);
  } else if(bop == bop_t::min) {
    return scalarop_t::make_min(dtype);
  } else if(bop == bop_t::max) {
    return scalarop_t::make_max(dtype);
  } else {
    throw std::runtime_error("Undefined bop detected");
  }
}

scalarop_t simple_scalarop_t::uop_to_scalarop(uop_t uop, dtype_t dtype) {
 if(uop == uop_t::identity) {
   return scalarop_t::make_identity(dtype);
 } else if(uop == uop_t::neg)  {
  return scalarop_t::make_neg(dtype);
 } else if (uop == uop_t::exp) {
  return scalarop_t::make_exp(dtype);
 } else if (uop == uop_t::log) {
  return scalarop_t::make_log(dtype);
 } else if (uop == uop_t::rcp) {
  return scalarop_t::make_rcp(dtype);
 } else if (uop == uop_t::conj) {
  return scalarop_t::make_conjugate(dtype);
 } else if (uop == uop_t::sqrt) {
  return scalarop_t::make_sqrt(dtype);
 } else if (uop == uop_t::relu) {
  return scalarop_t::make_relu(dtype);
 } else if (uop == uop_t::sigmoid) {
  return scalarop_t::make_sigmoid(dtype);
 } else {
   throw std::runtime_error("missing uop case...........");
 }
}

dtype_t simple_scalarop_t::get_inn_dtype(int which) const {
  if(is_scale()) {
    if(which != 0) {
      throw std::runtime_error("scale only has one input");
    }
    return get_scale().scale.dtype;
  }
  if(is_unary()) {
    if(which != 0) {
      throw std::runtime_error("scale only has one input");
    }
    return get_unary().scale.dtype;
  }
  if(is_binary()) {
    if(which < 0 || which > 1) {
      throw std::runtime_error("binary has two inputs");
    }
    if(which == 0) {
      return get_binary().lhs.scale.dtype;
    } else {
      return get_binary().rhs.scale.dtype;
    }
  }
  throw std::runtime_error("should not occur: missing case get_inn_dtype");
}

int simple_scalarop_t::num_inns() const {
  if(is_scale() || is_unary()) {
    return 1;
  }
  if(is_binary()) {
    return 2;
  }
  throw std::runtime_error("num inns: should not reach");
}

scalarop_t simple_scalarop_t::to_scalarop() const {
  if(is_scale()) {
    auto const& [bop, constant] = get_scale();
    return scalarop_t::combine(
      bop_to_scalarop(bop, constant.dtype),
      { 
        scalarop_t::make_identity(constant.dtype),
        scalarop_t::make_constant(constant)
      });
  } else if(is_unary()) {
    return unary_to_scalarop(get_unary());
  } else if(is_binary()) {
    auto const& [bop, lhs, rhs] = get_binary();
    return scalarop_t::combine(
      bop_to_scalarop(bop, lhs.scale.dtype),
      { unary_to_scalarop(lhs), unary_to_scalarop(rhs) });
  } else {
    throw std::runtime_error("simple_op not unary and not binary?");
  }
}

scalarop_t list_simple_scalarop_t::to_scalarop() const {
  vector<scalarop_t> results;
  auto get = [&](int which, dtype_t dtype) {
    if(which >= 0) {
      return scalarop_t::make_arg(which, dtype);
    } else {
      // which=-1 -> 0
      // which=-2 -> 1
      // ... and so on
      return results.at(-1*which-1);
    }
  };
  for(auto const& [simple_op, args]: ops) {
    if(simple_op.is_unary() || simple_op.is_scale()) {
      dtype_t inn_dtype = simple_op.get_inn_dtype(0);
      auto result = scalarop_t::combine(
        simple_op.to_scalarop(),
        { get(args[0], inn_dtype) });
      results.push_back(result);
    }
    else if(simple_op.is_binary()) {
      dtype_t lhs_dtype = simple_op.get_inn_dtype(0);
      dtype_t rhs_dtype = simple_op.get_inn_dtype(1);
      auto result = scalarop_t::replace_arguments(
        simple_op.to_scalarop(),
        { get(args[0], lhs_dtype), get(args[1], rhs_dtype) });
      results.push_back(result);
    } else {
      throw std::runtime_error("simple_op not unary and not binary?");
    }
  }
  return results.back();
}

struct list_maker_t {
  // This says that when you set
  // this node to have a new_id, make sure to tell
  // out_id that this the the new which_arg.
  struct info_t {
    int out_id;
    int which_arg;
    scalar_ns::node_t const* node;
  };

  // Return the units to recurse on, if we were able to make progress

  optional<vector<info_t>> add_op(scalar_ns::node_t const* node)
  {
    int ret_id = ops.size();

    auto maybe = pop_to_simple_scalarop(node);
    if(!maybe) {
      return std::nullopt;
    }

    auto const& [scalarop, child_nodes] = maybe.value();
    
    ops.push_back(list_simple_scalarop_t::op_t {
      .op = scalarop
    });

    vector<info_t> children;
    for(int arg = 0; arg != child_nodes.size(); ++arg) {
      children.push_back(info_t {
        .out_id = ret_id,
        .which_arg = arg,
        .node = child_nodes[arg]
      });
    }
    return children;
  }

  // op_ids are in the vector, which has 0,1,2,3,...
  // Map 0 -> -1, 1 -> -2, 2 -> -3, 3 -> -4 and so on
  static int op_id_to_arg_id(int op_id) {
    return -1*(1 + op_id);
  }
  static int inverse_op_id_to_arg_id(int inverse_id) {
    return -1*inverse_id - 1;
  }

  bool recurse(info_t info) {
    // DOUT("recurse");
    auto const& [prev_id, which_arg, node] = info;

    // the base case of the recursion
    if(node->op.is_hole()) {
      auto [arg, _] = node->op.get_hole();
      ops[prev_id].args[which_arg] = arg;
      return true;
    }

    auto maybe_children = add_op(node);
    if(!maybe_children) {
      return false;
    }
    auto const& new_children = maybe_children.value();
    int new_op_id = new_children[0].out_id;
    ops[prev_id].args[which_arg] = op_id_to_arg_id(new_op_id);
    for(auto const& new_info: new_children) {
      if(!recurse(new_info)) {
        return false;
      }
    }

    return true;
  }

  vector<list_simple_scalarop_t::op_t> ops;
};

// TODO: check if this is correct
dtype_t list_simple_scalarop_t::max_dtype() const {
  std::unordered_map<dtype_t, int> dtype_to_weight;
  dtype_to_weight.insert({dtype_t::f16, 0});
  dtype_to_weight.insert({dtype_t::f32, 1});
  dtype_to_weight.insert({dtype_t::f64, 2});
  dtype_to_weight.insert({dtype_t::c64, 3});

  dtype_t max_dtype = dtype_t::f16;
  for (auto const& [op, args] : ops) {
    if (op.is_scale()) {
      auto const& [bop, scale] = op.get_scale();
      if (dtype_to_weight.at(scale.dtype) > dtype_to_weight.at(max_dtype)) {
        max_dtype = scale.dtype;
      }
    } else if (op.is_unary()) {
      auto const& [scale, uop] = op.get_unary();
      if (dtype_to_weight.at(scale.dtype) > dtype_to_weight.at(max_dtype)) {
        max_dtype = scale.dtype;
      }
    } else if (op.is_binary()) {
      auto const& [bop, lhs, rhs] = op.get_binary();
      if (dtype_to_weight.at(lhs.scale.dtype) > dtype_to_weight.at(max_dtype)) {
        max_dtype = lhs.scale.dtype;
      }
      if (dtype_to_weight.at(rhs.scale.dtype) > dtype_to_weight.at(max_dtype)) {
        max_dtype = rhs.scale.dtype;
      }
    }
  }
  return max_dtype;
}

void list_simple_scalarop_t::print(std::ostream& out) const {
  auto print_arg = [&](int arg) {
    if(arg >= 0) {
      out << "input " << arg;
    } else {
      out << "tmp " << list_maker_t::op_id_to_arg_id(arg);
    }
  };

  for(int i = 0; i != ops.size(); ++i) {
    auto const& [op, args] = ops[i];
    out << i << ": " << op;
    out << " { arg0: ";
    print_arg(args[0]);
    if(op.num_inns() == 2) {
      out << ", arg1: ";
      print_arg(args[1]);
    }
    out << " }" << std::endl;
  }
}

optional<list_simple_scalarop_t>
list_simple_scalarop_t::make(scalarop_t const& scalarop)
{
  // TODO: we put here to filter out all the identities
  // check if this is the best place for it
  if (scalarop.is_identity()){
    op_t op = op_t {
      .op = simple_scalarop_t::unary_t {
        .scale = scalar_t::one(scalarop.out_dtype()),
        .op = simple_scalarop_t::uop_t::identity
      },
    };
    op.args[0] = 0;
    return list_simple_scalarop_t {
      .ops = {op}
    };
  }
  scalar_ns::node_t const* node = &scalarop.node;

  list_maker_t maker;
  auto maybe_start = maker.add_op(node);
  if(!maybe_start) {
    return std::nullopt;
  }
  for(auto const& children: maybe_start.value()) {
    maker.recurse(children);
  }

  // At this point, maker.ops has what we need but in reverse order.
  // So (1) reverse the ops and (2) relabel all of the negative args
  // to point to the correct op
  list_simple_scalarop_t ret {
    .ops = maker.ops
  };
  std::reverse(ret.ops.begin(), ret.ops.end());

  vector<int> idxs = vector_iota<int>(ret.ops.size());
  std::reverse(idxs.begin(), idxs.end());

  auto fix = [&](int id) {
    if(id >= 0) {
      // this is an external input, nothing to do
      return id;
    }
    int prev_op_id = list_maker_t::inverse_op_id_to_arg_id(id);
    int new_op_id = idxs[prev_op_id];
    // ^ Given ret.ops.size(), there is probably a faster way to
    //   get new_op_id from prev_op_id, but this is quicker to write
    //   and less error prone
    return list_maker_t::op_id_to_arg_id(new_op_id);
  };

  for(auto& [op,args]: ret.ops) {
    args[0] = fix(args[0]);
    if(op.is_binary()) {
      args[1] = fix(args[1]);
    }
  }

  for (auto r: ret.ops){
    DOUT(r.op);
  }

  return ret;
}

std::ostream& operator<<(std::ostream& out, simple_scalarop_t const& op)
{
  if(op.is_scale()) {
    auto const& [bop, scale] = op.get_scale();
    out << bop << "(x," << scale << ")";
  } else if(op.is_unary()) {
    auto const& [scale, uop] = op.get_unary();
    out << scale << " * " << uop << "(x)";
  } else if(op.is_binary()) {
    auto const& [bop, lhs, rhs] = op.get_binary();
    out << bop << "(" << 
      lhs.scale << " * " << lhs.op << "(x), " <<
      rhs.scale << " * " << rhs.op << "(y))";
  } else {
    throw std::runtime_error("should not reach: simple scalarop print");
  }

  return out;
}

std::ostream& operator<<(std::ostream& out, simple_scalarop_t::uop_t const& op)
{
  switch(op) {
    case simple_scalarop_t::identity: out << "identity"; break;
    case simple_scalarop_t::neg:      out << "neg";      break;
    case simple_scalarop_t::sqrt:     out << "sqrt";     break;
    case simple_scalarop_t::conj:     out << "conj";     break;
    case simple_scalarop_t::rcp:      out << "rcp";      break;
    case simple_scalarop_t::sigmoid:  out << "sigmoid";  break;
    case simple_scalarop_t::log:      out << "log";      break;
    case simple_scalarop_t::exp:      out << "exp";      break;
    case simple_scalarop_t::relu:     out << "relu";     break;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, simple_scalarop_t::bop_t const& op)
{
  switch (op){
    case simple_scalarop_t::add: std::cout << "add"; break;
    case simple_scalarop_t::mul: std::cout << "mul"; break;
    case simple_scalarop_t::min: std::cout << "min"; break;
    case simple_scalarop_t::max: std::cout << "max"; break;
  }
  return out;
}
