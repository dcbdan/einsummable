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

int matgraph_t::insert_adds(vector<int> items) {
  if(items.size() < 2) {
    throw std::runtime_error("invalid insert_adds input");
  }

  while(items.size() != 1) {
    int n = items.size() / 2;
    int r = items.size() % 2;
    vector<int> next_up;
    next_up.reserve(n + r);
    if(r == 1) {
      next_up.push_back(items.back());
    }
    for(int i = 0; i != n; ++i) {
      next_up.push_back(insert_ewb(scalar_join_t::add, items[2*i], items[2*i+1]));
    }
    items = next_up;
  }
  return items[0];
}

optional<int> matgraph_t::node_t::inn0() const
{
  using std::get;

  if(is_matmul()) {
    auto const& _op = get<matmul_t>(op);
    return optional<int>(_op.lhs);
  } else if(is_ew()) {
    auto const& _op = get<ew_t>(op);
    return optional<int>(_op.inn);
  } else if(is_ewb()) {
    auto const& _op = get<ewb_t>(op);
    return optional<int>(_op.lhs);
  } else if(is_input()) {
    return optional<int>();
  } else if(is_ones()) {
    return optional<int>();
  } else {
    throw std::runtime_error("should not reach");
  }
}

optional<int> matgraph_t::node_t::inn1() const
{
  using std::get;

  if(is_matmul()) {
    auto const& _op = get<matmul_t>(op);
    return optional<int>(_op.rhs);
  } else if(is_ew()) {
    return optional<int>();
  } else if(is_ewb()) {
    auto const& _op = get<ewb_t>(op);
    return optional<int>(_op.rhs);
  } else if(is_input()) {
    return optional<int>();
  } else if(is_ones()) {
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
set<int> matgraph_t::node_t::inns_set() const {
  auto is = inns();
  return set<int>(is.begin(), is.end());
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

  for(auto const& inn: node.inns_set()) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

bool matgraph_t::node_t::is_einsummable() const
{
  return is_matmul() || is_ew() || is_ewb();
}

bool matgraph_t::node_t::is_matmul() const
{
  return std::holds_alternative<matmul_t>(op);
}
bool matgraph_t::node_t::is_ew() const
{
  return std::holds_alternative<ew_t>(op);
}
bool matgraph_t::node_t::is_ewb() const
{
  return std::holds_alternative<ewb_t>(op);
}
bool matgraph_t::node_t::is_input() const
{
  return std::holds_alternative<input_t>(op);
}
bool matgraph_t::node_t::is_ones() const
{
  return std::holds_alternative<ones_t>(op);
}


// TODO: implement this without recursion, following the looping structure
//       of compile.
vector<int> matgraph_t::backprop(int out, vector<int> weights)
{
  // It should be the case that every node in nodeset will have
  // a gradient computed for it.
  // If there are no connected paths from a weight to the out,
  // then this will fail at some point, which is fine
  // because such cases refer to zerod gradients, which is
  // silly.
  set<int> nodeset = compute_nodeset({out}, weights, true);

  backprop_state_t state {
    .grads = {},
    .self = *this,
    .nodeset = std::move(nodeset)
  };

  // the base case of the recursion
  state.start(out);

  // This will recursively get call backprop_state_t::operator[]
  // to compute all the gradients in the nodeset.
  // (The recursion shouldn't be an issue since the graph probably isn't
  //  that big)
  vector<int> ret;
  ret.reserve(weights.size());
  for(auto const& weight: weights) {
    ret.push_back(state[weight]);
  }
  return ret;
}

int matgraph_t::backprop_state_t::operator[](int id) {
  if(grads.count(id) > 0) {
    // This gradient has already been computed
    return grads.at(id);
  }
  if(nodeset.count(id) == 0) {
    // This should not happen
    throw std::runtime_error("this id isn't in the nodeset");
  }

  // Let e1, e2, ..., en be all edges going out from node id
  // that are in the nodeset.
  // Then g[id] = sum_i { grad_at[e(i)] "*" this[e(i).out] }
  //
  // Note: out = f(inn, inn) would have two edges from inn -> out

  auto const& node = self.nodes[id];
  vector<out_edge_t> out_edges = get_out_edges(id);

  vector<int> terms;
  terms.reserve(out_edges.size());
  for(auto const& [out, which_inn]: out_edges) {
    auto const& out_grad = (*this)[out]; // recurse
    terms.push_back(
      self.build_grad_term(out, which_inn, out_grad)
    );
  }

  if(terms.size() == 0) {
    throw std::runtime_error("no terms. Is there a path?");
  }

  int ret;
  if(terms.size() == 1) {
    ret = terms[0];
  } else {
    ret = self.insert_adds(terms);
  }
  grads.insert({id, ret});
  return ret;
};

void matgraph_t::backprop_state_t::start(int out_id)
{
  grads.insert({out_id, out_id});
}

vector<matgraph_t::backprop_state_t::out_edge_t>
matgraph_t::backprop_state_t::get_out_edges(int id) const
{
  auto const& outs = self.nodes[id].outs;

  vector<out_edge_t> ret;
  ret.reserve(2*outs.size());

  for(auto const& out: outs) {
    if(nodeset.count(out) > 0) {
      auto inns = self.nodes[out].inns();
      for(int which_inn = 0; which_inn != inns.size(); ++which_inn)
      {
        auto const& inn = inns[which_inn];
        if(inn == id) {
          ret.push_back(out_edge_t {
            .out = out,
            .which_inn = which_inn
          });
        }
      }
    }
  }
  return ret;
}

set<int> matgraph_t::compute_nodeset(
  vector<int> const& upps,
  vector<int> const& dwns,
  bool include_upps_dwns) const
{
  // Walk down the graph collecting all nodes
  // touched from the upps
  set<int> upp_dwn;
  for(auto const& upp: upps) {
    for(auto const& inn: nodes[upp].inns_set()) {
      upp_dwn.insert(inn);
    }
  }
  {
    set<int> pending = upp_dwn;
    while(pending.size() > 0) {
      set<int> next_up;
      for(auto const& upp: pending) {
        for(auto const& inn: nodes[upp].inns_set()) {
          upp_dwn.insert(inn);
          next_up.insert(inn);
        }
      }
      pending = std::move(next_up);
    }
  }

  // Walk up the graph collection all nodes
  // touched from the dwns
  set<int> dwn_upp;
  for(auto const& dwn: dwns) {
    for(auto const& out: nodes[dwn].outs) {
      dwn_upp.insert(out);
    }
  }
  {
    set<int> pending = dwn_upp;
    while(pending.size() > 0) {
      set<int> next_up;
      for(auto const& dwn: pending) {
        for(auto const& out: nodes[dwn].outs) {
          dwn_upp.insert(out);
          next_up.insert(out);
        }
      }
      pending = std::move(next_up);
    }
  }

  // Set ret = intersection of dwn_upp and upp_dwn
  set<int> ret;
  for(auto const& id: upp_dwn) {
    if(dwn_upp.count(id) > 0) {
      ret.insert(id);
    }
  }

  if(include_upps_dwns) {
    for(auto const& upp: upps) {
      ret.insert(upp);
    }
    for(auto const& dwn: dwns) {
      ret.insert(dwn);
    }
  }

  return ret;
}

int matgraph_t::build_grad_term(
  int node_id,
  int which_inn,
  int node_grad)
{
  using std::get;

  auto const& node = nodes[node_id];
  auto const& op = node.op;
  auto inns = node.inns();
  if(which_inn >= inns.size()) {
    throw std::runtime_error("invalid inn in build graph term");
  }

  if(node.is_matmul()) {
    if(which_inn == 0) {
      return build_grad_term_matmul_lhs(get<matmul_t>(op), node_grad);
    } else {
      return build_grad_term_matmul_rhs(get<matmul_t>(op), node_grad);
    }
  } else if(node.is_ewb()) {
    if(which_inn == 0) {
      return build_grad_term_ewb_lhs(get<ewb_t>(op), node_grad);
    } else {
      return build_grad_term_ewb_rhs(get<ewb_t>(op), node_grad);
    }
  } else if(node.is_ew()) {
    return build_grad_term_ew_inn(get<ew_t>(op), node_grad);
  } else {
    throw std::runtime_error("should not reach");
  }
}

int matgraph_t::build_grad_term_matmul_lhs(
  matgraph_t::matmul_t const& matmul,
  int node_grad)
{
  auto const& [t_lhs, lhs, t_rhs, rhs] = matmul;

  // Compute: d(LR)/d(L) "*" node_grad = R "*" node_grad

  // node_grad shape: ik
  // return shape == lhs shape
  if(t_lhs) {
    // lhs shape: ji
    if(t_rhs) {
      // rhs shape: kj
      // kj,ik->ji
      return insert_matmul_tt(rhs, node_grad);
    } else {
      // rhs shape: jk
      // jk,ik->ji
      return insert_matmul_st(rhs, node_grad);
    }
  } else {
    // lhs shape: ij
    if(t_rhs) {
      // rhs shape: kj
      // ik,kj->ij
      return insert_matmul_ss(node_grad, rhs);
    } else {
      // rhs shape: jk
      // ik,jk->ij
      return insert_matmul_st(node_grad, rhs);
    }
  }
}

int matgraph_t::build_grad_term_matmul_rhs(
  matgraph_t::matmul_t const& matmul,
  int node_grad)
{
  auto const& [t_lhs, lhs, t_rhs, rhs] = matmul;
  // Compute: d(LR)/d(R) "*" node_grad = L "*" node_grad

  // node_grad shape: ik
  // return shape == rhs shape
  if(t_rhs) {
    // rhs shape: kj
    if(t_lhs) {
      // lhs shape: ji
      // ik,ji->kj
      return insert_matmul_tt(node_grad, lhs);
    } else {
      // lhs shape: ij
      // ik,ij->kj
      return insert_matmul_ts(node_grad, lhs);
    }
  } else {
    // rhs shape: jk
    if(t_lhs) {
      // lhs shape: ji
      // ji,ik->jk
      return insert_matmul_ss(lhs, node_grad);
    } else {
      // lhs shape: ij
      // ij,ik->jk
      return insert_matmul_ts(lhs, node_grad);
    }
  }
}

int matgraph_t::build_grad_term_ewb_lhs(
  matgraph_t::ewb_t const& ewb,
  int node_grad)
{
  auto const& [op, lhs, rhs] = ewb;
  if(op == scalar_join_t::add) {
    // d(lhs + rhs) /d lhs .* node_grad
    return node_grad;
  } else if(op == scalar_join_t::sub) {
    // d(lhs - rhs)/d lhs .* node_grad
    return node_grad;
  } else if(op == scalar_join_t::mul) {
    // d(lhs * rhs) /d lhs .* node_grad
    return insert_ewb(scalar_join_t::mul, rhs, node_grad);
  } else if(op == scalar_join_t::min) {
    throw std::runtime_error("build grad term ewb rhs: min");
  } else if(op == scalar_join_t::max) {
    throw std::runtime_error("build grad term ewb rhs: max");
  } else {
    throw std::runtime_error("build grad term ewb rhs: should not reach");
  }
}

// enum class scalar_join_t { add, sub, mul, relu, negate, min, max };

int matgraph_t::build_grad_term_ewb_rhs(
  matgraph_t::ewb_t const& ewb,
  int node_grad)
{
  auto const& [op, lhs, rhs] = ewb;
  if(op == scalar_join_t::add) {
    // d(lhs + rhs) /d rhs .* node_grad
    return node_grad;
  } else if(op == scalar_join_t::sub) {
    // d(lhs - rhs)/d rhs .* node_grad
    return insert_ew(scalar_join_t::negate, node_grad);
  } else if(op == scalar_join_t::mul) {
    // d(lhs * rhs) /d rhs .* node_grad
    return insert_ewb(scalar_join_t::mul, lhs, node_grad);
  } else if(op == scalar_join_t::min) {
    throw std::runtime_error("build grad term ewb rhs: min");
  } else if(op == scalar_join_t::max) {
    throw std::runtime_error("build grad term ewb rhs: max");
  } else {
    throw std::runtime_error("build grad term ewb rhs: should not reach");
  }
}

int matgraph_t::build_grad_term_ew_inn(
  matgraph_t::ew_t const& ew,
  int node_grad)
{
  auto const& [op, _] = ew;

  if(op == scalar_join_t::negate) {
    return insert_ew(scalar_join_t::negate, node_grad);
  } else if(op == scalar_join_t::relu) {
    throw std::runtime_error("build grad term ew: no relu deriv scalar");
  } else {
    throw std::runtime_error("build grad term ew: should not reach");
  }
}

void matgraph_t::node_t::print() const
{
  using std::get;

  auto const& [d0,d1] = out_shape;
  std::cout << "shape: " << d0 << ", " << d1 << std::endl;

  std::cout << "op: ";
  if(is_matmul()) {
    auto const& [t_lhs, _, t_rhs, _1] = get<matmul_t>(op);
    std::cout << "matmul[";
    if(t_lhs) {
      std::cout << "ji";
    } else {
      std::cout << "ij";
    }
    std::cout << ",";
    if(t_rhs) {
      std::cout << "kj";
    } else {
      std::cout << "jk";
    }
    std::cout << "->ik]";
  } else if(is_ew()) {
    auto const& [scalar_op, _] = get<ew_t>(op);
    std::cout << "ew ";
    if(scalar_op == scalar_join_t::negate) {
      std::cout << "negate";
    }
  } else if(is_ewb()) {
    auto const& [scalar_op, _0, _1] = get<ewb_t>(op);
    std::cout << "ewb";
  } else if(is_input()) {
    std::cout << "input";
  } else if(is_ones()) {
    std::cout << "ones";
  } else {
    throw std::runtime_error("should not reach");
  }
  std::cout << std::endl;

  std::cout << "inns:  " << inns() << std::endl;
  std::cout << "outs:  " << vector<int>(outs.begin(), outs.end()) << std::endl;
}

void matgraph_t::print() const
{
  for(int id = 0; id != nodes.size(); ++id) {
    std::cout << "id: " << id << std::endl;
    nodes[id].print();
    std::cout << std::endl;
  }
}

tuple<graph_t, map<int, int> >
matgraph_t::compile() const
{
  vector<int> outs;
  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];
    if(node.outs.size() == 0) {
      outs.push_back(id);
    }
  }

  return compile(outs);
}

tuple<graph_t, map<int, int> >
matgraph_t::compile(vector<int> const& saves) const
{
  vector<int> pending;
  map<int, int> remaining;
  {
    vector<int> inns;
    for(int id = 0; id != nodes.size(); ++id) {
      auto const& node = nodes[id];
      if(node.is_input()) {
        inns.push_back(id);
      }
    }


    set<int> nodeset = compute_nodeset(saves, inns, true);

    // set remaining, but make sure not to add
    // any input into remaining as it will be in pending
    for(auto const& id: nodeset) {
      auto const& node = nodes[id];
      if(!vector_has(inns, id)) {
        remaining.insert({id, node.inns_set().size()});
      }
    }

    pending = std::move(inns);
    // now the inputs are in pending
  }

  graph_t ret;
  map<int, int> matgraph_to_graph;

  while(pending.size() != 0) {
    vector<int> next_up;
    for(auto const& mid: pending) {
      node_t const& node = nodes[mid];

      if(node.is_ones()) {
        // ones nodes are not added to the return graph
        // and instead should be absorbed directly
        // into einsummable ops
      } else {
        int gid;
        if(node.is_einsummable()) {
          // translate_op will attempt to absorb ones
          // into the einsummable expression.
          auto [einsummable, matgraph_inns] = translate_node(node);
          vector<int> graph_inns;
          graph_inns.reserve(matgraph_inns.size());
          for(auto const& m_inn: matgraph_inns) {
            int const& g_inn = matgraph_to_graph.at(m_inn);
            graph_inns.push_back(g_inn);
          }
          gid = ret.insert_einsummable(einsummable, graph_inns);
        } else if(node.is_input()) {
          auto const& [d0,d1] = node.out_shape;
          gid = ret.insert_input({d0,d1});
        } else {
          throw std::runtime_error("should not reach");
        }

        if(vector_has(saves, mid)) {
          int gsaveid = ret.insert_formation(gid, true);
          matgraph_to_graph.insert({mid, gsaveid});
        } else {
          matgraph_to_graph.insert({mid, gid});
        }
      }

      // Only add to next_up if it in the remaining set
      // and therefore was part of nodeset and
      // has all the inputs.
      for(auto const& out: node.outs) {
        if(remaining.count(out) > 0) {
          int& cnt = remaining.at(out);
          cnt -= 1;
          if(cnt == 0) {
            next_up.push_back(out);
            remaining.erase(out);
          }
        }
      }
    }
    pending = std::move(next_up);
  }

  // Make sure all nodes have been accounted for
  if(remaining.size() != 0) {
    throw std::runtime_error("Not all nodes in nodeset has been accounted for");
  }

  // Make sure the save nodes are handled correctly.
  // Note: including a ones as a save node will trigger an error here.
  for(auto const& save_mid: saves) {
    if(matgraph_to_graph.count(save_mid) == 0) {
      throw std::runtime_error("A save node is not in the output graph");
    }
    int gid = matgraph_to_graph.at(save_mid);
    auto const& gnode = ret.nodes[gid];
    if(!gnode.op.is_save()) {
      throw std::runtime_error("A save node was not properly saved");
    }
  }

  return {ret, matgraph_to_graph};
}

tuple<einsummable_t, vector<int>>
matgraph_t::translate_node(node_t const& node) const
{
  if(!node.is_einsummable()) {
    throw std::runtime_error("translate node input must be einsummable");
  }

  // TODO: Everything to get to this method is designed to allow for
  //       absorbing ones. But this method doesn't bother implementing
  //       that. Maybe it will be necessary later.
  {
    vector<int> inns = node.inns();
    for(auto const& inn: inns) {
      auto const& inn_node = nodes[inn];
      if(inn_node.is_ones()) {
        throw std::runtime_error("absorbing ones not implemented");
      }
    }
  }

  if(node.is_matmul()) {
    auto const& [t_lhs, id_lhs, t_rhs, id_rhs] = std::get<matmul_t>(node.op);

    auto const& [lhs_d0, lhs_d1] = nodes[id_lhs].out_shape;
    auto const& [rhs_d0, rhs_d1] = nodes[id_rhs].out_shape;

    if(t_lhs) {
      uint64_t const& dj = lhs_d0;
      uint64_t const& di = lhs_d1;
      if(t_rhs) {
        uint64_t const& dk = rhs_d0;
        return {einsummable_t::from_matmul_tt(di, dj, dk), {id_lhs, id_rhs} };
      } else {
        uint64_t const& dk = rhs_d1;
        return {einsummable_t::from_matmul_ts(di, dj, dk), {id_lhs, id_rhs} };
      }
    } else {
      uint64_t const& di = lhs_d0;
      uint64_t const& dj = lhs_d1;
      if(t_rhs) {
        uint64_t const& dk = rhs_d0;
        return {einsummable_t::from_matmul_st(di, dj, dk), {id_lhs, id_rhs} };
      } else {
        uint64_t const& dk = rhs_d1;
        return {einsummable_t::from_matmul_ss(di, dj, dk), {id_lhs, id_rhs} };
      }
    }
  } else if(node.is_ewb()) {
    auto const& [op, id_lhs, id_rhs] = std::get<ewb_t>(node.op);
    auto const& [d0,d1] = node.out_shape;
    einsummable_t e {
      .join_shape = {d0, d1},
      .inns = { {0, 1}, {0, 1} },
      .out_rank = 2,
      .join = op,
      .castable = castable_t::add,
    };
    return {e, {id_lhs, id_rhs}};
  } else if(node.is_ew()) {
    auto const& [op, id_inn] = std::get<ew_t>(node.op);
    auto const& [d0,d1] = node.out_shape;
    einsummable_t e {
      .join_shape = {d0, d1},
      .inns = { {0, 1} },
      .out_rank = 2,
      .join = op,
      .castable = castable_t::add,
    };
    return {e, {id_inn}};
  } else {
    throw std::runtime_error("should not reach");
  }
}


