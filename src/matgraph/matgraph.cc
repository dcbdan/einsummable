#include "matgraph.h"

int matgraph_t::insert_ew(scalar_join_t op, int inn)
{
  auto const& out_shape = nodes[inn].out_shape;
  if(!is_unary_scalar_join(op)) {
    throw std::runtime_error("exepects unary scalar join op");
  }
  return insert(
    op_t(ew_t {
      .op = op,
      .inn = inn
    }),
    out_shape);
}

int matgraph_t::insert_ewb(scalar_join_t op, int lhs, int rhs)
{
  if(!is_binary_scalar_join(op)) {
    throw std::runtime_error("exepects binary scalar join op");
  }

  auto const& [i, j ] = nodes[lhs].out_shape;
  auto const& [i_,j_] = nodes[rhs].out_shape;
  if(i != i_ || j != j_) {
    throw std::runtime_error("ewb: invalid inputs");
  }

  return insert(
    op_t(ewb_t {
      .op = op,
      .lhs = lhs,
      .rhs = rhs
    }),
    {i,j}
  );
}

int matgraph_t::insert_matmul_ss(int lhs, int rhs)
{
  auto const& [i,j]  = nodes[lhs].out_shape;
  auto const& [j_,k] = nodes[rhs].out_shape;
  if(j != j_) {
    throw std::runtime_error("matmul_ss: invalid inputs");
  }

  return insert(
    op_t(matmul_t {
      .t_lhs = false,
      .lhs = lhs,
      .t_rhs = false,
      .rhs = rhs
    }),
    {i,k});
}

int matgraph_t::insert_matmul_ts(int lhs, int rhs)
{
  auto const& [j,i]  = nodes[lhs].out_shape;
  auto const& [j_,k] = nodes[rhs].out_shape;
  if(j != j_) {
    throw std::runtime_error("matmul_ss: invalid inputs");
  }

  return insert(
    op_t(matmul_t {
      .t_lhs = true,
      .lhs = lhs,
      .t_rhs = false,
      .rhs = rhs
    }),
    {i,k});
}

int matgraph_t::insert_matmul_st(int lhs, int rhs)
{
  auto const& [i,j]  = nodes[lhs].out_shape;
  auto const& [k,j_] = nodes[rhs].out_shape;
  if(j != j_) {
    throw std::runtime_error("matmul_ss: invalid inputs");
  }

  return insert(
    op_t(matmul_t {
      .t_lhs = false,
      .lhs = lhs,
      .t_rhs = true,
      .rhs = rhs
    }),
    {i,k});
}

int matgraph_t::insert_matmul_tt(int lhs, int rhs)
{
  auto const& [j,i]  = nodes[lhs].out_shape;
  auto const& [k,j_] = nodes[rhs].out_shape;
  if(j != j_) {
    throw std::runtime_error("matmul_ss: invalid inputs");
  }

  return insert(
    op_t(matmul_t {
      .t_lhs = true,
      .lhs = lhs,
      .t_rhs = true,
      .rhs = rhs
    }),
    {i,k});
}

int matgraph_t::insert_input(uint64_t d0, uint64_t d1)
{
  return insert(
    op_t(input_t {}),
    {d0, d1});
}

int matgraph_t::insert_ones(uint64_t d0, uint64_t d1)
{
  return insert(
    op_t(ones_t {}),
    {d0, d1});
}

std::optional<int> matgraph_t::node_t::inn0() const
{
  using std::holds_alternative;
  using std::get;

  if(holds_alternative<matmul_t>(op)) {
    auto const& _op = get<matmul_t>(op);
    return optional<int>(_op.lhs);
  } else if(holds_alternative<ew_t>(op)) {
    auto const& _op = get<ew_t>(op);
    return optional<int>(_op.inn);
  } else if(holds_alternative<ewb_t>(op)) {
    auto const& _op = get<ewb_t>(op);
    return optional<int>(_op.lhs);
  } else if(holds_alternative<input_t>(op)) {
    return optional<int>();
  } else if(holds_alternative<ones_t>(op)) {
    return optional<int>();
  } else {
    throw std::runtime_error("should not reach");
  }
}

std::optional<int> matgraph_t::node_t::inn1() const
{
  using std::holds_alternative;
  using std::get;

  if(holds_alternative<matmul_t>(op)) {
    auto const& _op = get<matmul_t>(op);
    return optional<int>(_op.rhs);
  } else if(holds_alternative<ew_t>(op)) {
    return optional<int>();
  } else if(holds_alternative<ewb_t>(op)) {
    auto const& _op = get<ewb_t>(op);
    return optional<int>(_op.rhs);
  } else if(holds_alternative<input_t>(op)) {
    return optional<int>();
  } else if(holds_alternative<ones_t>(op)) {
    return optional<int>();
  } else {
    throw std::runtime_error("should not reach");
  }
}

vector<int> matgraph_t::node_t::inns() const {
  auto i0 = inn0();
  if(i0) {
    auto i1 = inn1();
    if(i1) {
      return {i0.value(), i1.value()};
    } else {
      return {i0.value()};
    }
  } else {
    return {};
  }
}

int matgraph_t::insert(matgraph_t::op_t op, tuple<uint64_t, uint64_t> out_shape)
{
  int ret = nodes.size();

  nodes.push_back(node_t {
    .out_shape = out_shape,
    .op = op,
    .outs = set<int>()
  });

  auto const& node = nodes.back();

  for(auto const& inn: node.inns()) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

// TODO
vector<int> backprop(int out, vector<int> weights)
{
  return {};
}

// TODO
graph_t compile(matgraph_t const& matgraph)
{
  return graph_t();
}
