#include "update.h"

static
einsummable_t
vanilla_make_einsummable(
  dtype_t dtype,
  vector<uint64_t> const& shape)
{
  scalarop_t grad_update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(dtype),
      scalarop_t::make_scale("learning_rate", dtype)
    }
  );

  int rank = shape.size();
  return einsummable_t(
    shape,
    { vector_iota<int>(rank), vector_iota<int>(rank) },
    rank,
    grad_update);
}

static
vector<tuple<int, fill_t>>
vanilla_update_weights(
  dtype_t dtype,
  graph_t& graph,
  vector<tuple<int, int>>& ret,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  ret.reserve(ret.size() + weight_ids.size());
  for(auto [weight, grad]: vector_zip(weight_ids, grad_ids)) {
    einsummable_t e = vanilla_make_einsummable(dtype, graph.out_shape(weight));
    int updated_weight = graph.insert_einsummable(e, {weight, grad});
    ret.emplace_back(weight, updated_weight);
  }

  return {};
}

static
void vanilla_update_vars(
  dtype_t dtype,
  map<string, scalar_t>& vars)
{
  // put a default learning rate if none is already in vars
  vars.insert({"learning_rate", scalar_t(dtype, "1e-3")});
}

static
void adamw_update_vars(
  dtype_t dtype,
  dtype_t min_precision,
  int iter,
  map<string, scalar_t>& vars)
{
  if(vars.count("beta1") == 0 ||
     vars.count("beta2") == 0 ||
     vars.count("eta") == 0)
  {
    throw std::runtime_error("missing vars provided");
  }

  scalar_t one = scalar_t::one(dtype);
  scalar_t one_mp = scalar_t::one(min_precision);

  vars.insert_or_assign("beta1_complement", one - vars["beta1"]);
  vars.insert_or_assign("beta2_complement", one - vars["beta2"]);

  scalarop_t power = scalarop_t::make_power(iter, min_precision);

  vars.insert_or_assign("beta1tt", one_mp / (one_mp - power.eval(vars["beta1"].convert(min_precision))));
  vars.insert_or_assign("beta2tt", one_mp / (one_mp - power.eval(vars["beta2"].convert(min_precision))));

  vars.insert({"eps", scalar_t(min_precision, "1e-8")});

  auto& eta = vars["eta"];
  eta = eta.convert(min_precision);
}

static
int adamw_insert_einsummable_ew(
  graph_t& graph,
  scalarop_t join,
  vector<int> const& inns)
{
  vector<uint64_t> shape = graph.out_shape(inns[0]);
  int rank = shape.size();

  einsummable_t e(
    shape,
    vector<vector<int>>(inns.size(), vector_iota<int>(rank)),
    rank,
    join);

  return graph.insert_einsummable(e, inns);
}

static
vector<tuple<int, fill_t>>
adamw_update_weights(
  dtype_t dtype,
  dtype_t min_precision,
  graph_t& graph,
  vector<tuple<int, int>>& ret,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  int n = weight_ids.size();

  if(n == 0) {
    return {};
  }

  scalarop_t up_prec   = scalarop_t::make_convert_dtype(dtype, min_precision);
  scalarop_t down_prec = scalarop_t::make_convert_dtype(min_precision, dtype);

  auto up_prec_scale = [&](string var) {
    return scalarop_t::combine(
      scalarop_t::make_scale(var, min_precision),
      { up_prec });
  };

  scalarop_t grad_update = scalarop_t::combine(
    scalarop_t::make_sub(min_precision),
    {
      up_prec,
      scalarop_t::make_scale("eta", min_precision)
    });
  grad_update = scalarop_t::combine(
    down_prec,
    { grad_update });

  scalarop_t beta1_portion = scalarop_t::combine(
    scalarop_t::make_add(dtype),
    {
      scalarop_t::make_scale("beta1", dtype),
      scalarop_t::make_scale("beta1_complement", dtype)
    });

  scalarop_t scale_square = scalarop_t::combine(
    scalarop_t::make_scale("beta2_complement", dtype),
    { scalarop_t::make_square(dtype) });

  scalarop_t beta2_portion = scalarop_t::combine(
    scalarop_t::make_add(dtype),
    {
      scalarop_t::make_scale("beta2", dtype),
      scale_square
    });

  scalarop_t sqrt_plus_eps = scalarop_t::combine(
    scalarop_t::make_add(min_precision),
    {
      scalarop_t::make_sqrt(min_precision),
      scalarop_t::make_variable("eps", min_precision)
    });

  scalarop_t beta1t_scale = up_prec_scale("beta1tt");
  scalarop_t beta2t_scale = up_prec_scale("beta2tt");

  vector<int> m_ids;
  vector<int> v_ids;

  m_ids.reserve(n);
  v_ids.reserve(n);
  for(auto const& w_id: weight_ids) {
    auto shape = graph.out_shape(w_id);
    m_ids.push_back(graph.insert_input(shape, dtype));
    v_ids.push_back(graph.insert_input(shape, dtype));
  }

  ret.reserve(ret.size() + n*3);
  for(int i = 0; i != n; ++i) {
    int const& w_id = weight_ids[i];
    int const& g_id = grad_ids[i];
    int const& m_id = m_ids[i];
    int const& v_id = v_ids[i];

    int m_new = adamw_insert_einsummable_ew(graph, beta1_portion, {m_id, g_id});
    int v_new = adamw_insert_einsummable_ew(graph, beta2_portion, {v_id, g_id});

    int mm = adamw_insert_einsummable_ew(graph, beta1t_scale, { m_new });
    int vv = adamw_insert_einsummable_ew(graph, beta2t_scale, { v_new });
    vv = adamw_insert_einsummable_ew(graph, sqrt_plus_eps, { vv });

    int xx = adamw_insert_einsummable_ew(graph, scalarop_t::make_div(min_precision), {mm, vv});

    int w_new = adamw_insert_einsummable_ew(graph, grad_update, {w_id, xx});

    ret.emplace_back(m_id, m_new);
    ret.emplace_back(v_id, v_new);
    ret.emplace_back(w_id, w_new);
  }

  vector<tuple<int, fill_t>> inits;
  inits.reserve(n*2);
  for(int i = 0; i != n; ++i) {
    int const& m_id = m_ids[i];
    int const& v_id = v_ids[i];
    inits.emplace_back(
      m_id,
      fill_t::make_constant(scalar_t::zero(dtype), graph.out_shape(m_id)));
    inits.emplace_back(
      v_id,
      fill_t::make_constant(scalar_t::zero(dtype), graph.out_shape(v_id)));
  }

  return inits;
}

vector<tuple<int, fill_t>>
update_weights(
  updater_desc_t const& t,
  graph_t& graph,
  vector<tuple<int, int>>& old_news,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  if(t.is_vanilla()) {
    return vanilla_update_weights(t.dtype, graph, old_news, weight_ids, grad_ids);
  } else if(t.is_adamw()) {
    auto const& a = t.get_adamw();
    return adamw_update_weights(
      t.dtype, a.min_precision,
      graph, old_news, weight_ids, grad_ids);
  } else {
    throw std::runtime_error("update_weights invalid desc");
  }
}

void increment_vars(
  updater_desc_t const& t,
  int iter,
  map<string, scalar_t>& vars)
{
  if(t.is_vanilla()) {
    return vanilla_update_vars(t.dtype, vars);
  } else if(t.is_adamw()) {
    auto const& a = t.get_adamw();
    return adamw_update_vars(t.dtype, a.min_precision, iter, vars);
  } else {
    throw std::runtime_error("update_weights invalid desc");
  }
}

