#include "graph.h"

vector<int> graph_t::backprop(int out, vector<int> weights) {
  if(dtype_is_complex(out_dtype(out))) {
    throw std::runtime_error(
      "cannot do backprop wrt complex loss; using wirtinger simplificaiton");
  }

  // Get nodes which values affect output of the graph
  set<int> nodeset = compute_nodeset({out}, weights, true);

  vector<int> ordered_nodeset = reverse_order_nodeset(nodeset);
  if(ordered_nodeset[0] != out) {
    throw std::runtime_error("invalid reverse_order_nodeset result");
  }

  backprop_state_t state {
    .grads = {},
    .self = *this,
    .nodeset = std::move(nodeset)
  };

  state.start(out);

  for(int i = 1; i != ordered_nodeset.size(); ++i) {
    int const& id = ordered_nodeset[i];
    state.insert(id);
  }

  vector<int> grads;
  grads.reserve(weights.size());
  for(auto const& weight : weights) {
    backprop_tensor_t grad = state.grads.at(weight);
    if(grad.is_constant()) {
      grads.push_back(insert_fill(grad.get_fill()));
    } else {
      grads.push_back(grad.get_id());
    }
  }

  for(int i = 0; i != grads.size(); ++i) {
    int const& w = weights[i];
    int const& g = grads[i];
    if(out_dtype(w) != out_dtype(g)) {
      throw std::runtime_error("incorrect dtype of grad");
    }
    if(out_shape(w) != out_shape(g)) {
      throw std::runtime_error("incorrect shape of grad");
    }
  }

  return grads;
}

graph_t::backprop_tensor_t::backprop_tensor_t()
  : op(-1)
{}

graph_t::backprop_tensor_t::backprop_tensor_t(int id)
  : op(id)
{}

graph_t::backprop_tensor_t::backprop_tensor_t(fill_t const& fill)
  : op(fill)
{}

graph_t::backprop_tensor_t
graph_t::backprop_tensor_t::backprop_tensor_t::ones(
  dtype_t const& dtype,
  vector<uint64_t> const& shape)
{
  scalar_t value;
  if(dtype_is_complex(dtype)) {
    if(dtype != dtype_t::c64) {
      throw std::runtime_error("not supported complex");
    }
    value = scalar_t(std::complex<float>(1.0, 0.0));
  } else {
    value = scalar_t(dtype, "1.0");
  }

  return backprop_tensor_t(fill_t {
    .value = value,
    .shape = shape
  });
}

graph_t::backprop_tensor_t
graph_t::backprop_tensor_t::backprop_tensor_t::zeros(
  dtype_t const& dtype,
  vector<uint64_t> const& shape)
{
  return backprop_tensor_t(fill_t {
    .value = scalar_t::zero(dtype),
    .shape = shape
  });
}

int const& graph_t::backprop_tensor_t::get_id() const {
  return std::get<int>(op);
}

fill_t const& graph_t::backprop_tensor_t::get_fill() const {
  return std::get<fill_t>(op);
}

scalar_t graph_t::backprop_tensor_t::get_constant() const {
  return get_fill().value;
}

bool graph_t::backprop_tensor_t::is_constant() const {
  return std::holds_alternative<fill_t>(op);
}

bool graph_t::backprop_tensor_t::is_constant_of(scalar_t v) const {
  if(is_constant()) {
    return get_fill().value == v;
  }
  return false;
}

bool graph_t::backprop_tensor_t::is_zeros() const {
  if(is_constant()) {
    scalar_t const& v = get_fill().value;
    return scalar_t::zero(v.dtype) == v;
  }
  return false;
}

bool graph_t::backprop_tensor_t::is_ones() const {
  if(is_constant()) {
    auto const& [scalar, _] = get_fill();

    // scalar_t::one is not valid for complex values, so use this
    // 1.0 value in the context of backprop
    if(dtype_is_complex(scalar.dtype)) {
      if(scalar.dtype != dtype_t::c64) {
        throw std::runtime_error("this complex dtype is not supported");
      }
      std::complex<float> v(1.0,0.0);
      return scalar_t(v) == scalar;
    }
    return scalar_t::one(scalar.dtype) == scalar;
  }
  return false;
}

dtype_t graph_t::backprop_tensor_t::dtype(graph_t& self) const {
  if(is_constant()) {
    return get_fill().value.dtype;
  } else {
    int const& id = get_id();
    return self.nodes[id].op.out_dtype();
  }
}

vector<uint64_t> graph_t::backprop_tensor_t::shape(graph_t& self) const {
  if(is_constant()) {
    return get_fill().shape;
  } else {
    int const& id = get_id();
    return self.nodes[id].op.out_shape();
  }
}

void graph_t::backprop_state_t::start(int out_id)
{
  auto const& op = self.nodes[out_id].op;
  backprop_tensor_t tensor =
    backprop_tensor_t::ones(op.out_dtype(), op.out_shape());
  grads.insert({out_id, tensor});
}

vector<graph_t::backprop_state_t::out_edge_t>
graph_t::backprop_state_t::get_out_edges(int id) const
{
  auto const& outs = self.nodes[id].outs;

  vector<out_edge_t> ret;
  ret.reserve(2*outs.size());

  for (auto const& out : outs) {
    if (nodeset.count(out) > 0) {
      auto inns = self.nodes[out].inns;
      for (int which_inn = 0; which_inn != inns.size(); ++which_inn)
      {
        auto const& inn = inns[which_inn];
        if (inn == id) {
          ret.emplace_back(out_edge_t {
            .out = out,
            .which_inn = which_inn
          });
        }
      }
    }
  }

  return ret;
}

void graph_t::backprop_state_t::insert(int id)
{
  if(grads.count(id) > 0 ) {
    throw std::runtime_error("this grad has already been inserted");
  }
  if(nodeset.count(id) == 0) {
    throw std::runtime_error("This id is not in the nodeset");
  }

  auto const& node = self.nodes[id];
  vector<out_edge_t> out_edges = get_out_edges(id);
  dtype_t dtype = node.op.out_dtype();

  vector<backprop_tensor_t> terms;
  terms.reserve(out_edges.size());
  for(auto const& [out, which_inn] : out_edges) {
    // building grad term for out with respect to this id
    backprop_tensor_t const& out_grad = grads.at(out);
    backprop_tensor_t term = self.build_grad_term(out, which_inn, out_grad);
    if(term.dtype(self) != dtype) {
      throw std::runtime_error("invalid term dtype during backprop");
    }
    terms.push_back(term);
  }

  if(terms.size() == 0) {
    throw std::runtime_error("No terms, no compute path");
  }

  backprop_tensor_t ret;
  if(terms.size() == 1) {
    ret = terms[0];
  } else {
    ret = self.insert_adds(terms);
  }

  grads.insert({id, ret});
}

graph_t::backprop_tensor_t
graph_t::build_grad_term(int id, int which_inn, backprop_tensor_t grad_id)
{
  auto const& node = nodes[id];
  auto const& op = node.op;
  auto const& inns = node.inns;

  if(which_inn < 0 || which_inn >= inns.size()) {
    throw std::runtime_error("Invalid which_inn in build graph term");
  }

  if(op.is_input()) {
    throw std::runtime_error("should not happen: grad at input");
  } else if(op.is_formation()) {
    // this node, at the graph_t level here, is just an identity operation.
    return grad_id;
  } else if(op.is_complexer()) {
    return build_grad_term_complexer(grad_id);
  } else if(op.is_squeezer()) {
    auto const& inn_shape = op.get_squeezer().inn_shape;
    return build_grad_term_squeezer(inn_shape, grad_id);
  } else if(op.is_fill()) {
    // fill is just constant values that don't depend on anything
    return backprop_tensor_t::zeros(op.out_dtype(), op.out_shape());
  } else if(op.is_select()) {
    return build_grad_term_select(
      op.get_select(), which_inn, grad_id);
  } else if(op.is_einsummable()) {
    return build_grad_term_einsummable(
      op.get_einsummable(), id, inns, which_inn, grad_id);
  } else {
    throw std::runtime_error("should not reach: missing graph type");
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_einsummable(
  einsummable_t const& e,
  int out_id,
  vector<int> const& inn_ids,
  int which_inn,
  backprop_tensor_t grad_id)
{
  int num_inn = inn_ids.size();

  if(e.has_broadcast()) {
    // Given ijk->ijkl,
    // form ijkl->ijk
    vector<int> inns;
    int out_rank;
    {
      set<int> bmodes = e.get_broadcast_modes();

      string letters(e.out_rank, ' ');
      std::iota(letters.begin(), letters.end(), 'a');

      string const& inn = letters;

      string out(e.out_rank - bmodes.size(), ' ');
      auto iter = out.begin();
      for(int i = 0; i != e.out_rank; ++i) {
        if(bmodes.count(i) == 0) {
          *iter++ = letters[i];
        }
      }
      if(iter != out.end()) {
        throw std::runtime_error("invalid iter end state");
      }

      string str = inn + "->" + out;

      auto [inns_, out_rank_] = einsummable_t::parse_str(str);
      inns = inns_[0];
      out_rank = out_rank_;
    }

    backprop_tensor_t fixed_grad_id = backprop_tensor_aggregate(grad_id, inns, out_rank);

    if(e.is_broadcast()) {
      // If this is just a broadcast, we have the answer
      return fixed_grad_id;
    } else {
      // This einsummable is a compute and then a broadcast,
      // recurse to the compute part.
      return build_grad_term_einsummable(
        e.remove_broadcast(),
        out_id,
        inn_ids,
        which_inn,
        fixed_grad_id);
    }
  }

  if(!e.has_aggregation()) {
    // no broadcast but an aggregation -> elementwise
    return build_grad_term_ew(e, inn_ids, which_inn, grad_id);
  }

  if(num_inn == 1 && e.join.is_identity()) {
    // Fix out_id so that if there is an outgoing formation, that gets used
    {
      auto const& out_node = nodes[out_id];
      for(auto const& out_out_id: out_node.outs) {
        auto const& out_out_node = nodes[out_out_id];
        if(out_out_node.op.is_formation()) {
          out_id = out_out_id;
          break;
        }
      }
    }

    auto const& castable = e.castable.value();
    if(castable == castable_t::add) {
      return build_grad_term_reduction_add(
        e.join_shape, e.inns[0], e.out_rank, grad_id);
    } else if(castable == castable_t::mul) {
      return build_grad_term_reduction_mul(
        e.join_shape, e.inns[0], e.out_rank,
        out_id, inn_ids[0], grad_id);
    } else if(castable == castable_t::max || castable == castable_t::min) {
      return build_grad_term_reduction_maxmin(
        e.join_shape, e.inns[0], e.out_rank,
        out_id, inn_ids[0], grad_id);
    } else {
      throw std::runtime_error("missing castable in reduction backprop");
    }
  } else if(e.is_contraction()) {
    // This is a contraction, so multiply the grad_id with either the
    // left or the right input
    return build_grad_term_contraction(e, inn_ids, which_inn, grad_id);
  }

  // One option would be to split the einsummable into
  //   (1) a reduction, (2) an elementwise op
  // Then return
  //   build_grad_term_einsummable(
  //     e_ew,
  //     inn_ids,
  //     which_inn,
  //     build_grad_term_einsummable(e_reduction, ?, 0, grad_id));
  // However, there is no inn_id to the reduction portion since it
  // is never formed.
  //
  // The solution seems to be either (a) don't form these non-standard "compound"
  // einsummables and (b) special case standard "compound" einsummables such
  // as contraction.

  throw std::runtime_error(
    "not implemented: einsummable characterized by \n"
    "no broadcast, with aggregation, not a contraction, not a reduction");
}

graph_t::backprop_tensor_t
graph_t::backprop_tensor_aggregate(
  graph_t::backprop_tensor_t const& tensor,
  vector<int> const& inn,
  int out_rank)
{
  einsummable_t e = einsummable_t::aggregate(
    tensor.shape(*this),
    inn, out_rank,
    tensor.dtype(*this),
    castable_t::add);

  if(tensor.is_constant()) {
    vector<uint64_t> out_shape = e.out_shape();
    scalar_t value = tensor.get_constant();
    uint64_t nelem_inn = product(e.join_shape);
    uint64_t nelem_out = product(out_shape);
    double multiplier(nelem_inn / nelem_out);
    value *= multiplier;
    return backprop_tensor_t(fill_t {
      .value = value,
      .shape = out_shape
    });
  } else {
    int const& id = tensor.get_id();
    return backprop_tensor_t(insert_einsummable_for_backprop(e, { id }));
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_contraction(
  einsummable_t const& e,
  vector<int> const& inn_ids,
  int which_inn,
  backprop_tensor_t grad)
{
  if(which_inn != 0 && which_inn != 1) {
    throw std::runtime_error("contraction has two inputs..");
  }

  dtype_t dtype = e.out_dtype();
  if(dtype_is_complex(dtype)) {
    // TODO
    throw std::runtime_error("grad_term_contraction with complex dtype: not implemented");
  }

  if(grad.is_zeros()) {
    vector<uint64_t> shape =
      which_inn == 0 ?
      e.inn_shape(0) :
      e.inn_shape(1) ;
    return backprop_tensor_t::zeros(dtype, shape);
  }

  if(grad.is_constant()) {
    //  Suppose we have this contraction
    //    ij,jk->ik
    //  where the rhs is a constant of value v != 0.0
    //
    //  Out[i,k] = Sum_j Lhs[i,j] * Rhs[j,k]
    //           = Sum_j v * Lhs[i,j]
    //  This is einsummable
    //    ij->ik
    //  where join_op = lambda x0: v*x0

    auto const& value = grad.get_fill().value;

    string new_str;
    vector<uint64_t> new_inn_shape, new_out_shape;
    int new_inn_id;
    auto [_, inn_strs] = e.str_terms();
    if(which_inn == 0) {
      new_inn_shape = e.inn_shape(1);
      new_inn_id = inn_ids[1];
      new_out_shape = e.inn_shape(0);
      new_str = inn_strs[1] + "->" + inn_strs[0];
    } else {
      new_inn_shape = e.inn_shape(0);
      new_inn_id = inn_ids[0];
      new_out_shape = e.inn_shape(1);
      new_str = inn_strs[0] + "->" + inn_strs[1];
    }

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      new_out_shape, new_inns, { new_inn_shape });

    // Note: if value == 1.0, this should get simplified to the identity function
    scalarop_t join = scalarop_t::make_scale(value);
    einsummable_t new_e(new_join_shape, new_inns, new_out_rank, join, e.castable);

    int ret_id = insert_einsummable_for_backprop(new_e, {new_inn_id});
    return backprop_tensor_t(ret_id);
  }

  int const& grad_id = grad.get_id();
  string new_str;
  int new_l_id, new_r_id;
  vector<uint64_t> new_l_shape, new_r_shape, new_o_shape;
  auto [out_str, inn_strs] = e.str_terms();
  if(which_inn == 0) {
    new_l_id = grad_id;
    new_r_id = inn_ids[1];
    new_str = out_str + "," + inn_strs[1] + "->" + inn_strs[0];
    new_l_shape = e.out_shape();
    new_r_shape = e.inn_shape(1);
    new_o_shape = e.inn_shape(0);
  } else { // which_inn == 1
    new_l_id = inn_ids[0];
    new_r_id = grad_id;
    new_str = inn_strs[0] + "," + out_str + "->" + inn_strs[1];
    new_l_shape = e.inn_shape(0);
    new_r_shape = e.out_shape();
    new_o_shape = e.inn_shape(1);
  }

  auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
  vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
    new_o_shape, new_inns, { new_l_shape, new_r_shape });
  einsummable_t new_e(new_join_shape, new_inns, new_out_rank, e.join, e.castable);

  return backprop_tensor_t(insert_einsummable_for_backprop(new_e, {new_l_id, new_r_id}));
}

// example: (yhat - y)**2 => 2(yhat - y) * node_grad
//          -------------    -----------
//          ^ op             ^ deri_op
//                           -----------------------
//                           ^ join_op
//          (here derivative wrt input yhat)


// example: f(x) => f'(x) * node_grad
//                  -----
//                  ^ deri_op
//                  -----------------
//                  ^ join_op
//
// For op = inn -> out:
//   If neither constant:
//     f'(inn) * node_grad
//     inn,out -> inn
//   If just deri_op is a constant:
//     deri_op_v * node_grad
//     out -> inn
//   If node_grad is a constant:
//     f'(x) * grad_v
//     inn -> inn
//   If both are constants:
//     v( = deri_op_v * grad_v)
//     constant of shape inn
//
// Note: for ensummable \x0,x1->x0*x1 and "ij,i->ij", the vjp requires a
//       a summation over j so not all of these are elementwise
graph_t::backprop_tensor_t
graph_t::build_grad_term_ew(
  einsummable_t const& e,
  vector<int> inn_ids,
  int which_inn,
  backprop_tensor_t grad)
{
  dtype_t out_dtype = e.out_dtype();
  dtype_t inn_dtype = e.inn_dtype(which_inn);

  // TODO: implement wirtinger derivatives properly using the grad update rule
  //   v* * (ds/d(z*)) + v * (ds/dz)*
  //
  // For now, detect just x*y otherwise throw an error
  if(dtype_is_complex(inn_dtype) || dtype_is_complex(out_dtype)) {
    if(e.join == scalarop_t::make_mul(out_dtype)) {
      // ds/d(z*) is zero since multiplication is holomorphic and
      // ds/dz is the other arg so return v * (other_arg)*
      dtype_t const& dtype = out_dtype;
      scalarop_t new_join = scalarop_t::combine(
        scalarop_t::make_mul(dtype),
        {
          scalarop_t::make_identity(dtype),
          scalarop_t::make_conjugate(dtype)
        });
      if(which_inn != 0 && which_inn != 1) {
        throw std::runtime_error("only two args: ew complex mul special case");
      }
      int other_id = which_inn == 0 ? inn_ids[1] : inn_ids[0];

      int v_id;
      if(grad.is_constant()) {
        v_id = insert_fill(grad.get_fill());
      } else {
        v_id = grad.get_id();
      }

      auto [out_str, inn_strs] = e.str_terms();
      string const& new_out_str = inn_strs[which_inn];
      string const& new_lhs_str = out_str;
      string const& new_rhs_str = inn_strs[which_inn == 0 ? 1 : 0];
      string new_str = new_lhs_str + "," + new_rhs_str + "->" + new_out_str;

      vector<uint64_t> new_out_shape = e.inn_shape(which_inn);
      vector<uint64_t> new_lhs_shape = e.out_shape();
      vector<uint64_t> new_rhs_shape = e.inn_shape(which_inn == 0 ? 1 : 0);

      auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
      auto new_join_shape = einsummable_t::construct_join_shape(
        new_out_shape, new_inns,
        {new_lhs_shape, new_rhs_shape});
      einsummable_t new_e(
        new_join_shape, new_inns, new_out_rank, new_join, castable_t::add);

      int term_id = insert_einsummable_for_backprop(new_e, { v_id, other_id });
      return backprop_tensor_t(term_id);
    } else {
      throw std::runtime_error("grad_term_ew: not supporting complex ew except x*y");
    }
  }

  scalarop_t deri_op = e.join.derivative(which_inn);

  bool constant_deri_op = deri_op.is_constant();
  bool constant_grad = grad.is_constant();

  // Note: remember we need to return a tensor of inn_dtype

  auto [out_str, inn_strs] = e.str_terms();
  vector<vector<uint64_t>> inn_shapes = e.inn_shapes();
  vector<uint64_t> out_shape = e.out_shape();

  if(constant_deri_op && constant_grad) {
    scalar_t grad_constant = grad.get_constant();
    scalar_t deri_constant = deri_op.eval({});

    scalar_t v = scalarop_t::make_mul(out_dtype).eval({grad_constant, deri_constant});
    v = v.convert(inn_dtype);

    return backprop_tensor_t(fill_t {
      .value = v,
      .shape = inn_shapes[which_inn]
    });
  } else if(constant_deri_op) {
    scalar_t value = deri_op.eval({}).convert(out_dtype);

    scalarop_t new_join = scalarop_t::combine(
      scalarop_t::make_mul(out_dtype),
      vector<scalarop_t> {
        scalarop_t::make_constant(value),
        scalarop_t::make_identity(out_dtype)
      }
    );

    new_join = scalarop_t::combine(
      scalarop_t::make_convert_dtype(out_dtype, inn_dtype),
      vector<scalarop_t>{ new_join });

    if(new_join.is_constant()) {
      // It could be the case that the new_join is simplified to a constant
      // function. (For example, value == 0.0)
      scalar_t new_value = new_join.eval({});
      return backprop_tensor_t(fill_t {
        .value = new_value,
        .shape = inn_shapes[which_inn]
      });
    }

    string new_str = out_str + "->" + inn_strs[which_inn];
    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      inn_shapes[which_inn], new_inns, { out_shape });
    einsummable_t new_e(
      new_join_shape, new_inns, new_out_rank,
      new_join, castable_t::add);
    int term_id = insert_einsummable_for_backprop(new_e, { grad.get_id() });
    return backprop_tensor_t(term_id);
  } else if(constant_grad) {
    scalar_t value = grad.get_constant();

    scalarop_t new_join = scalarop_t::combine(
      scalarop_t::make_mul(out_dtype),
      vector<scalarop_t> {
        deri_op,
        scalarop_t::make_constant(value)
      }
    );

    new_join = scalarop_t::combine(
      scalarop_t::make_convert_dtype(out_dtype, inn_dtype),
      vector<scalarop_t>{ new_join });

    if(new_join.is_constant()) {
      // It could be the case that the new_join is simplified to a constant
      // function. (For example, value == 0.0)
      scalar_t new_value = new_join.eval({});
      return backprop_tensor_t(fill_t {
        .value = new_value,
        .shape = inn_shapes[which_inn]
      });
    }

    map<int,int> remap;
    int _cnt = 0;
    vector<string> actual_inn_strs;
    vector<int> new_inn_ids;
    vector<vector<uint64_t>> new_inn_shapes;
    for(int i = 0; i != inn_strs.size(); ++i) {
      if(new_join.is_used(i)) {
        actual_inn_strs.push_back(inn_strs[i]);
        new_inn_ids.push_back(inn_ids[i]);
        new_inn_shapes.push_back(inn_shapes[i]);
        remap.insert({i,_cnt++});
      }
    }

    new_join.remap_inputs(remap);

    string new_str = actual_inn_strs[0];
    for(int i = 1; i != actual_inn_strs.size(); ++i) {
      new_str += "," + actual_inn_strs[i];
    }
    new_str += "->" + inn_strs[which_inn];

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      inn_shapes[which_inn], new_inns, new_inn_shapes);
    einsummable_t new_e(
      new_join_shape, new_inns, new_out_rank,
      new_join, castable_t::add);
    int term_id = insert_einsummable_for_backprop(new_e, new_inn_ids);
    return backprop_tensor_t(term_id);
  } else {
    // Note: it's important that the identity op come first
    //       since we don't know what holes deri_op uses, just
    //       that a hole is used..
    scalarop_t new_join = scalarop_t::combine(
      scalarop_t::make_mul(out_dtype),
      vector<scalarop_t> {
        scalarop_t::make_identity(out_dtype),
        deri_op,
      }
    );

    new_join = scalarop_t::combine(
      scalarop_t::make_convert_dtype(out_dtype, inn_dtype),
      vector<scalarop_t>{ new_join });

    map<int,int> remap;
    int _cnt = 0;
    vector<string> actual_inn_strs;
    vector<int> new_inn_ids;
    vector<vector<uint64_t>> new_inn_shapes;
    if(new_join.is_used(0)) {
      actual_inn_strs.push_back(out_str);
      new_inn_ids.push_back(grad.get_id());
      new_inn_shapes.push_back(out_shape);
      remap.insert({0,_cnt++});
    }
    for(int i = 0; i != inn_strs.size(); ++i) {
      int which_hole = i + 1;
      if(new_join.is_used(which_hole)) {
        actual_inn_strs.push_back(inn_strs[i]);
        new_inn_ids.push_back(inn_ids[i]);
        new_inn_shapes.push_back(inn_shapes[i]);
        remap.insert({which_hole,_cnt++});
      }
    }

    new_join.remap_inputs(remap);

    if(actual_inn_strs.size() != new_join.num_inputs()) {
      throw std::runtime_error("invalid join op!");
    }

    string new_str = actual_inn_strs[0];
    for(int i = 1; i != actual_inn_strs.size(); ++i) {
      new_str += "," + actual_inn_strs[i];
    }
    new_str += "->" + inn_strs[which_inn];

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      inn_shapes[which_inn], new_inns, new_inn_shapes);
    einsummable_t new_e(new_join_shape, new_inns, new_out_rank,
      new_join, castable_t::add);
    int term_id = insert_einsummable_for_backprop(new_e, new_inn_ids);
    return backprop_tensor_t(term_id);
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_select(
  select_t const& select,
  int which_inn,
  backprop_tensor_t grad)
{
  // If grad is zeros, then just return zeros from the input shape
  if(grad.is_zeros()) {
    dtype_t const& dtype = grad.get_constant().dtype;
    return backprop_tensor_t::zeros(dtype, select.inn_shape(which_inn));
  }

  // If which_inn uses the full input tensor:
  bool uses_full_input = true;
  {
    for(auto const& sd: select.inn_regions[which_inn]) {
      if(sd.d_inn != sd.size) {
        uses_full_input = false;
        break;
      }
    }
  }
  if(uses_full_input) {
    if(grad.is_constant()) {
      return backprop_tensor_t(fill_t {
        .value = grad.get_constant(),
        .shape = select.inn_shape(which_inn)
      });
    }
    // Subset the gradient and return that
    int const& grad_id = grad.get_id();
    return backprop_tensor_t(insert_subset(
      select.wrt_output_inn_hrect(which_inn),
      grad_id));
  }

  // Only portion of the input region is getting selected by the output. All other
  // portions are to be set to zero.
  hrect_t inn_grad_hrect = select.wrt_input_inn_hrect(which_inn);
  vector<uint64_t> inn_shape = select.inn_shape(which_inn);
  int rank = inn_shape.size();
  partition_t partition = [&] {
    vector<partdim_t> pds;
    pds.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      auto const& [beg,end] = inn_grad_hrect[i];
      uint64_t const& size = inn_shape[i];
      vector<uint64_t> spans;
      if(beg != 0) {    spans.push_back(beg); }
      if(end != size) { spans.push_back(end); }
      spans.push_back(size);
      pds.push_back(partdim_t::from_spans(spans));
    }
    return partition_t(pds);
  }();

  dtype_t const& dtype = select.dtype;

  vector<int> block_shape = partition.block_shape();
  int num_blocks = product(block_shape);

  vector<int> inn_ids;
  inn_ids.reserve(num_blocks);

  using selectdim_t = select_t::selectdim_t;

  vector<vector<selectdim_t>> inn_regions;
  inn_regions.reserve(num_blocks);

  vector<int> index(rank);
  do {
    hrect_t block_hrect = partition.get_hrect(index);

    // Case 1: this block is the grad block
    //   Case 1a: grad is just a constant, so we need to fill the block with
    //            a constant of that value
    //   Case 1b: grad is not a constant, so we need to use those values
    // Case 2: this block is not the grad block, fill the block with zeros

    // Filling with a constant is mechanically
    // the same code, so separate those out
    optional<scalar_t> maybe_constant_fill = std::nullopt;
    if(block_hrect == inn_grad_hrect) {
      if(grad.is_constant()) {
        maybe_constant_fill = grad.get_constant();
      } else {
        // case 1b
      }
    } else {
      maybe_constant_fill = scalar_t::zero(dtype);
    }

    if(maybe_constant_fill) {
      // Case 1a, Case 2
      inn_regions.emplace_back();
      auto& inn_region = inn_regions.back();
      inn_region.reserve(rank);
      vector<uint64_t> block_shape;
      block_shape.reserve(rank);
      for(int i = 0; i != rank; ++i) {
        auto const& [beg,end] = block_hrect[i];
        uint64_t size = end-beg;
        inn_region.push_back(selectdim_t {
          .d_inn = size,
          .offset_inn = 0,
          .offset_out = beg,
          .size = size
        });
        block_shape.push_back(size);
      }

      inn_ids.push_back(insert_fill(fill_t {
        .value = maybe_constant_fill.value(),
        .shape = block_shape
      }));
    } else {
      // Case 1b
      vector<uint64_t> out_start = select.wrt_output_point(
        vector_mapfst(block_hrect),
        which_inn);
      vector<uint64_t> const& out_shape = select.out_shape;

      inn_regions.emplace_back();
      auto& inn_region = inn_regions.back();
      inn_region.reserve(rank);
      vector<uint64_t> block_shape;
      block_shape.reserve(rank);
      for(int i = 0; i != rank; ++i) {
        auto const& [beg,end] = block_hrect[i];
        uint64_t size = end-beg;
        auto const& o_beg = out_start[i];
        auto const& o_dim = out_shape[i];
        inn_region.push_back(selectdim_t {
          .d_inn = o_dim,
          .offset_inn = o_beg,
          .offset_out = beg,
          .size = size
        });
        block_shape.push_back(size);
      }

      int const& grad_id = grad.get_id();
      inn_ids.push_back(grad_id);
    }
  } while(increment_idxs(block_shape, index));

  select_t new_select(dtype, inn_shape, inn_regions);
  return backprop_tensor_t(insert(op_t(new_select), inn_ids));
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_complexer(
  graph_t::backprop_tensor_t grad)
{
  // The complexer op is a no op, basically...
  // if grad is complex, turn it real
  // if grad is real, turn it complex
  if(grad.is_constant()) {
    vector<uint64_t> shape = grad.get_fill().shape;
    scalar_t value = grad.get_constant();
    dtype_t const& dtype = value.dtype;
    if(dtype_is_real(dtype)) {
      // (v,v,v,v,v,v) -> ( (v,v), (v,v), (v,v) )
      if(dtype != dtype_t::f32) {
        throw std::runtime_error("complexer confusion");
      }
      float v = value.f32();
      std::complex<float> vv(v,v);

      if(shape.back() % 2 != 0) {
        throw std::runtime_error("odd number of last dims in complexer");
      }
      vector<uint64_t> complex_shape = shape;
      complex_shape.back() /= 2;

      return backprop_tensor_t(fill_t {
        .value = scalar_t(vv),
        .shape = complex_shape
      });
    } else {
      // ( (v,v), (v,v), (v,v) ) ->  (v,v,v,v,v,v)
      if(dtype != dtype_t::c64) {
        throw std::runtime_error("complexer confusion");
      }
      std::complex<float> vu = value.c64();
      if(vu.real() == vu.imag()) {
        float v = vu.real();
        vector<uint64_t> real_shape = shape;
        real_shape.back() *= 2;
        return backprop_tensor_t(fill_t {
          .value = scalar_t(v),
          .shape = real_shape
        });
      } else {
        // ( (v,u), (v,u), (v,u) ) cannot be represented as a constant
        // float tensor, so form it and convert it real
        int id = insert_fill(fill_t {
          .value = value,
          .shape = shape
        });
        return backprop_tensor_t(insert_to_real(id));
      }
    }
  } else {
    int const& id = grad.get_id();
    dtype_t dtype = out_dtype(id);
    if(dtype_is_real(dtype)) {
      return backprop_tensor_t(insert_to_complex(id));
    } else {
      return backprop_tensor_t(insert_to_real(id));
    }
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_squeezer(
  vector<uint64_t> const& inn_shape,
  graph_t::backprop_tensor_t grad)
{
  if(grad.is_constant()) {
    return backprop_tensor_t(fill_t {
      .value = grad.get_constant(),
      .shape = inn_shape
    });
  }

  return backprop_tensor_t(insert_squeezer(inn_shape, grad.get_id()));
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_reduction_add(
  vector<uint64_t> const& join_shape,
  vector<int> const& inn,
  int out_rank,
  graph_t::backprop_tensor_t grad)
{
  // This is easy, just do a broadcast in the other direction

  vector<uint64_t> inn_shape = einsummable_t::get_input_from_join_(
    inn,
    join_shape);

  if(grad.is_constant()) {
    scalar_t value = grad.get_constant();

    return backprop_tensor_t(fill_t {
      .value = value,
      .shape = inn_shape
    });
  }

  auto [old_out_str, old_inn_strs] = einsummable_t::make_str_terms({ inn }, out_rank);
  string const& old_inn_str = old_inn_strs[0];
  string new_str = old_out_str + "->" + old_inn_str;

  int const& id = grad.get_id();
  dtype_t dtype = out_dtype(id);

  auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
  einsummable_t e(
    inn_shape, new_inns, new_out_rank,
    scalarop_t::make_identity(dtype));

  return backprop_tensor_t(insert_einsummable_for_backprop(e, { id }));
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_reduction_mul(
  vector<uint64_t> const& join_shape,
  vector<int> const& inn,
  int out_rank,
  int out_id,
  int inn_id,
  graph_t::backprop_tensor_t grad)
{
  dtype_t dtype = grad.dtype(*this);

  vector<uint64_t> inn_shape = einsummable_t::get_input_from_join_(
    inn,
    join_shape);

  vector<uint64_t> out_shape(join_shape.begin(), join_shape.begin() + out_rank);


  auto [old_out_str, old_inn_strs] = einsummable_t::make_str_terms({ inn }, out_rank);
  string const& old_inn_str = old_inn_strs[0];

  // For ij->i, input X, output Y
  // Wij = Yi / Xij * ui
  scalarop_t base = scalarop_t::make_div(dtype);

  if(grad.is_constant()) {
    if(grad.is_zeros()) {
      return backprop_tensor_t::zeros(dtype, inn_shape);
    }
    scalar_t value = grad.get_constant();
    scalarop_t join = scalarop_t::combine(
      scalarop_t::make_mul(dtype),
      { base, scalarop_t::make_constant(value) });

    if(join.num_inputs() != 2) {
      throw std::runtime_error("expecting that all inputs are used");
    }

    string new_str = old_out_str + "," + old_inn_str + "->" + old_inn_str;

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    auto new_join_shape = einsummable_t::construct_join_shape(
      inn_shape,
      new_inns,
      { out_shape, inn_shape });

    einsummable_t e(new_join_shape, new_inns, new_out_rank, join);
    return backprop_tensor_t(insert_einsummable_for_backprop(e, { out_id, inn_id }));
  }

  scalarop_t join = scalarop_t::combine(
    scalarop_t::make_mul(dtype),
    { base, scalarop_t::make_arg(0, dtype) });

  if(join.num_inputs() != 3) {
    throw std::runtime_error("expecting that all inputs are used. ");
  }

  string new_str = old_out_str + "," + old_inn_str + "," + old_out_str + "->" + old_inn_str;

  auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
  auto new_join_shape = einsummable_t::construct_join_shape(
    inn_shape,
    new_inns,
    { out_shape, inn_shape, out_shape });

  int grad_id = grad.get_id();
  einsummable_t e(new_join_shape, new_inns, new_out_rank, join);
  return backprop_tensor_t(insert_einsummable_for_backprop(e, { out_id, inn_id, grad_id }));
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_reduction_maxmin(
  vector<uint64_t> const& join_shape,
  vector<int> const& inn,
  int out_rank,
  int out_id,
  int inn_id,
  graph_t::backprop_tensor_t grad)
{
  // Question: What is the backprop over max([v,v,v]) ?
  //   a) [1,0,0]
  //   b) [1/3,1/3,1/3]
  //   c) [1,1,1]
  // The issue is that all or any of the elements contribute to the maximum.
  // For option a, pick any of them
  // For option b and c, pick all of them
  // But option b normalizes to prevent increasing the backprop signal
  //
  // We choose option b because we can implemented. Option c is nice because
  // we don't have to add that many more ops... But I suspect that will hurt
  // us at some point.
  //
  // For ij->i, input X, output Y:
  //   which_max_ij = X_ij == Y_i
  //   sum_max_i    = Sum_j which_max_ij
  //   W_ij = (which_max_ij / sum_max_i) * u_i
  //
  // (The same holds if we are reducing with respect to minimum.)

  dtype_t dtype = grad.dtype(*this);

  vector<uint64_t> inn_shape = einsummable_t::get_input_from_join_(
    inn,
    join_shape);

  if(grad.is_constant() && grad.is_zeros()) {
    return backprop_tensor_t::zeros(dtype, inn_shape);
  }

  vector<uint64_t> out_shape(join_shape.begin(), join_shape.begin() + out_rank);

  auto [i_str, _inn_strs] = einsummable_t::make_str_terms({ inn }, out_rank);
  string const& ij_str = _inn_strs[0];

  int const& x_id = inn_id;
  auto const& x_shape = inn_shape;
  auto const& ij_shape = inn_shape;

  int const& y_id = out_id;
  auto const& y_shape = out_shape;
  auto const& i_shape = out_shape;

  auto make_ij_i = [&](scalarop_t const& join) {
    string str = ij_str + "," + i_str + "->" + ij_str;
    auto const& [e_inns,e_out_rank] = einsummable_t::parse_str(str);
    auto e_join_shape = einsummable_t::construct_join_shape(
      ij_shape,
      e_inns,
      { ij_shape, i_shape });
    return einsummable_t(e_join_shape, e_inns, e_out_rank, join);
  };

  auto make_ij_i_i = [&](scalarop_t const& join) {
    string str = ij_str + "," + i_str + "," + i_str + "->" + ij_str;
    auto const& [e_inns,e_out_rank] = einsummable_t::parse_str(str);
    auto e_join_shape = einsummable_t::construct_join_shape(
      ij_shape,
      e_inns,
      { ij_shape, i_shape, ij_shape });
    return einsummable_t(e_join_shape, e_inns, e_out_rank, join);
  };

  auto make_reduction = [&]() {
    return einsummable_t::aggregate(ij_shape, inn, out_rank, dtype, castable_t::add);
  };

  int which_max = insert_einsummable_for_backprop(
    make_ij_i(scalarop_t::make_is_equal(dtype)),
    { x_id, y_id });

  int sum_max = insert_einsummable_for_backprop(
    make_reduction(),
    { which_max });

  scalarop_t div = scalarop_t::make_div(dtype);

  if(grad.is_constant()) {
    scalar_t value = grad.get_constant();
    scalarop_t join = scalarop_t::combine(
      scalarop_t::make_mul(dtype),
      { div, scalarop_t::make_constant(value) });

    if(join.num_inputs() != 2) {
      throw std::runtime_error("expecting that all inputs are used");
    }

    einsummable_t e = make_ij_i(join);
    return backprop_tensor_t(
      insert_einsummable_for_backprop(e, { which_max, sum_max }));
  }

  scalarop_t join = scalarop_t::combine(
    scalarop_t::make_mul(dtype),
    { div, scalarop_t::make_arg(0, dtype) });

  if(join.num_inputs() != 3) {
    throw std::runtime_error("expecting that all inputs are used. ");
  }

  einsummable_t e = make_ij_i_i(join);
  return backprop_tensor_t(
    insert_einsummable_for_backprop(e, { which_max, sum_max, grad.get_id() }));
}

graph_t::backprop_tensor_t
graph_t::insert_adds(vector<backprop_tensor_t> const& items_)
{
  if(items_.size() == 0) {
    throw std::runtime_error("should not be summing empty list of tensors");
  }

  dtype_t dtype;
  vector<uint64_t> shape;
  int rank;
  {
    auto const& tensor = items_[0];
    dtype = tensor.dtype(*this);
    shape = tensor.shape(*this);
    rank = shape.size();
  }

  scalar_t sum = scalar_t::zero(dtype);
  vector<int> items;
  items.reserve(items_.size());
  for(auto const& tensor: items_) {
    if(tensor.is_constant()) {
      sum += tensor.get_constant();
    } else {
      items.push_back(tensor.get_id());
    }
  }

  if(items.size() == 0) {
    // In this case, all the terms were constants
    return backprop_tensor_t(fill_t {
      .value = sum,
      .shape = shape
    });
  }

  if(sum != scalar_t::zero(dtype)) {
    items.push_back(insert_fill(fill_t {
      .value = sum,
      .shape = shape
    }));
  }

  vector<int> is = vector_iota<int>(rank);
  vector<vector<int>> inns{ is, is };
  einsummable_t e(shape, inns, rank, scalarop_t::make_add(dtype));

  while(items.size() != 1) {
    int n = items.size() / 2;
    int r = items.size() % 2;
    vector<int> next_up;
    next_up.reserve(n + r);
    if(r == 1) {
      next_up.push_back(items.back());
    }
    for(int i = 0; i != n; ++i) {
      next_up.push_back(insert_einsummable_for_backprop(e, {items[2*i], items[2*i+1]}));
    }
    items = next_up;
  }

  return backprop_tensor_t(items[0]);
}

int graph_t::insert_einsummable_for_backprop(
  einsummable_t e,
  vector<int> inns)
{
  if(e.is_identity()) {
    // then inns.size() == 1
    return inns[0];
  }

  int ret = insert_einsummable(e, inns);

  if(e.has_aggregation()) {
    ret = insert_formation(ret, false);
  }

  return ret;
}

set<int> graph_t::compute_nodeset(
  vector<int> const& upps,
  vector<int> const& dwns,
  bool include_upps_dwns) const
{
  set<int> upp_dwn;
  for(auto const& upp: upps) {
    for(auto const& inn: nodes[upp].get_inns_set()) {
      upp_dwn.insert(inn);
    }
  }
  {
    set<int> pending = upp_dwn;
    while(pending.size() > 0) {
      set<int> next_up;
      for(auto const& upp: pending) {
        for(auto const& inn: nodes[upp].get_inns_set()) {
          upp_dwn.insert(inn);
          next_up.insert(inn);
        }
      }
      pending = std::move(next_up);
    }
  }

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

vector<int>
graph_t::reverse_order_nodeset(set<int> const& nodeset) const
{
  vector<int> ret;
  ret.reserve(nodeset.size());

  map<int, int> counts;
  vector<int> pending;
  for(int const& id: nodeset) {
    auto const& node = nodes[id];
    int& cnt = counts[id];
    for(int const& out_id: node.outs) {
      if(nodeset.count(out_id) > 0) {
        cnt++;
      }
    }
    if(cnt == 0) {
      pending.push_back(id);
    }
  }

  while(pending.size() > 0) {
    int id = pending.back();
    pending.pop_back();

    ret.push_back(id);

    auto const& node = nodes[id];
    for(int const& inn_id: node.get_inns_set()) {
      auto iter = counts.find(inn_id);
      if(iter != counts.end()) {
        int& cnt = iter->second;
        cnt--;
        if(cnt == 0) {
          pending.push_back(inn_id);
        }
      }
    }
  }

  if(ret.size() != nodeset.size()) {
    throw std::runtime_error("invalid size for reverse order return");
  }

  return ret;
}

