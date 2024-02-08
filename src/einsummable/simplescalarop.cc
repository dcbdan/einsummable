#include "simplescalarop.h"

namespace scalar_ns {

tuple<
  simple_scalarop_t::arg_t,
  node_t const*>
_pop_arg(node_t const* node)
{
  simple_scalarop_t::bop_t bop = simple_scalarop_t::bop_t::add;

  if(node->op.is_mul() || node->op.is_add()) {
    if(node->op.is_mul()) {
      bop = simple_scalarop_t::bop_t::mul;
    } else if(node->op.is_add()) {
      bop = simple_scalarop_t::bop_t::add;
    } else {
      throw std::runtime_error("should not reach");
    }

    node_t const* lhs = &(node->children[0]);
    node_t const* rhs = &(node->children[1]);

    if(lhs->op.is_constant() || rhs->op.is_constant())
    {
      scalar_t scalar = lhs->op.is_constant() ?
        lhs->op.get_constant()                :
        rhs->op.get_constant()                ;
      node_t const* child = lhs->op.is_constant() ?
        rhs                                       :
        lhs                                       ;
      simple_scalarop_t::arg_t arg {
        .scale_op = bop,
        .op = simple_scalarop_t::uop_t::identity,
        .scale = scalar
      };
      return { arg, child };
    }
  }

  simple_scalarop_t::arg_t arg {
    .scale_op = bop,
    .op = simple_scalarop_t::uop_t::identity,
    .scale = scalar_t::one(node->dtype)
  };

  return { arg, node };
}

optional<tuple<
  simple_scalarop_t::unary_t,
  node_t const*>>
_pop_to_unary(node_t const* node)
{
  // TODO
  return std::nullopt;
}

optional<tuple<
  simple_scalarop_t::binary_t,
  node_t const*,
  node_t const*>>
_pop_to_binary(node_t const* node)
{
  // TODO
  return std::nullopt;
}

optional<tuple<
  simple_scalarop_t,
  vector<node_t const*> > >
pop_to_simple_scalarop(node_t const* node)
{
  using value_t = tuple<simple_scalarop_t, vector<node_t const*>>;

  // TODO: put any special cases here

  {
    auto maybe = _pop_to_binary(node);
    if(maybe) {
      auto const& [binary, lhs, rhs] = maybe.value();
      return value_t {
        simple_scalarop_t { .op = binary },
        { lhs, rhs }
      };
    }
  }

  {
    auto maybe = _pop_to_unary(node);
    if(maybe) {
      auto const& [unary, child] = maybe.value();
      return value_t {
        simple_scalarop_t { .op = unary },
        { child }
      };
    }
  }

  return std::nullopt;
}

}

scalarop_t simple_scalarop_t::to_scalarop() const {
  // TODO
  throw std::runtime_error("not implemented");
}

scalarop_t list_simple_scalarop_t::to_scalarop() const {
  // TODO
  throw std::runtime_error("not implemented");
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

optional<list_simple_scalarop_t>
list_simple_scalarop_t::make(scalarop_t const& scalarop)
{
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

  return ret;
}

