#include "../src/base/setup.h"
#include "../src/base/args.h"

#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/dbuffer.h"

#include "modules.h"

#include <fstream>

using tensor_t     = graph_writer_t::tensor_t;
using full_dim_t   = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

int _get_label(
  graph_t const& graph,
  vector<int> const& checkpoints,
  int gid)
{
  auto iter = std::find(checkpoints.begin(), checkpoints.end(), gid);
  if(iter != checkpoints.end()) {
    return std::distance(checkpoints.begin(), iter);
  }

  auto const& node = graph.nodes[gid];
  if(node.outs.size() == 0) {
    return checkpoints.size();
  }

  int ret = -1;
  for(auto const& out_gid: node.outs) {
    ret = std::max(ret, _get_label(graph, checkpoints, out_gid));
  }

  return ret;
}

int get_label(
  graph_t const& graph,
  vector<int> const& checkpoints,
  int gid)
{
//  auto iter = std::find(checkpoints.begin(), checkpoints.end(), gid);
//  if(iter != checkpoints.end()) {
//    return std::distance(checkpoints.begin(), iter) + 1;
//  }

  return _get_label(graph, checkpoints, gid);
}

struct ffinfo_t {
  graph_t graph;
  int x;
  int y;
  vector<int> ows;
  vector<int> nws;
  int loss;
  vector<int> checkpoints;
  map<int, int> forward_labels;
};

ffinfo_t make_ff_simple(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim,
  int num_weight = 3)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});
  tensor_t y = writer.input({batch, dim});

  num_weight = std::max(num_weight, 3);

  vector<tensor_t> ws;
  ws.push_back(writer.input({dim, hidden}));
  for(int i = 0; i != num_weight - 2; ++i) {
    ws.push_back(writer.input({hidden, hidden}));
  }
  ws.push_back(writer.input({hidden, dim}));

  scalarop_t f = scalarop_t::make_relu(x.get_dtype());

  vector<int> checkpoints;
  tensor_t z = x;
  for(int i = 0; i != ws.size(); ++i) {
    tensor_t& w = ws[i];
    z = writer.ew(f, writer.matmul(z, w));
    if(i % 2 == 0) {
      checkpoints.push_back(z.get_id());
    }
  }

  scalarop_t difference_squared = scalarop_t::combine(
    scalarop_t::make_square(z.get_dtype()),
    { scalarop_t::make_sub(z.get_dtype()) });
  tensor_t loss = writer.ew("ij,ij->ij", difference_squared, z, y);
  loss.save_inplace();

  map<int, int> forward_labels;
  for(int gid = 0; gid != writer.get_graph().nodes.size(); ++gid) {
    int label = get_label(writer.get_graph(), checkpoints, gid);
    forward_labels.insert({gid, label});
  }

  vector<int> _f = vector_iota<int>(writer.get_graph().nodes.size());
  set<int> forward_ids(_f.begin(), _f.end());

  vector<tensor_t> grads = writer.backprop(loss, ws);

  scalarop_t update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(),
      scalarop_t::make_scale("learning_rate")
    });

  vector<tensor_t> update_ws;
  for(int i = 0; i != grads.size(); ++i) {
    tensor_t& w = ws[i];
    tensor_t& g = grads[i];
    update_ws.push_back(writer.ew("ij,ij->ij", update, w, g));
    update_ws.back().save_inplace();
  }

  return ffinfo_t {
    .graph = writer.get_graph(),
    .x = x.get_id(),
    .y = y.get_id(),
    .ows = vector_from_each_method(ws, int, get_id),
    .nws = vector_from_each_method(update_ws, int, get_id),
    .loss = loss.get_id(),
    .checkpoints = checkpoints,
    .forward_labels = forward_labels
  };
}

ffinfo_t make_llama_7B(
  uint64_t batch,
  uint64_t seqlen,
  int n_layers = -1)
{
  set_default_dtype(dtype_t::f16);
  dtype_t dtype = default_dtype();

  model_args_t margs = model_args_t::llama(1, batch);

  margs.max_seq_len = seqlen;

  if(n_layers >= 0) {
    margs.n_layers = std::min(margs.n_layers, n_layers);
  }

  graph_writer_t writer;
  optional<int> lora_rank = 8;
  transformer_t model(&writer, margs, 0, lora_rank);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));
  tensor_t labels = writer.input(
    vector<uint64_t>{margs.batch_size, margs.vocab_size},
    dtype);

  tensor_t predictions = model.forward(embeddings);

  // Compute the loss
  //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
  //   Loss = sum_n (l{n}) / N
  // Note, shift by c for numerical stability;
  //   where c{n} = max_c v{n,c}
  tensor_t loss;
  {
    tensor_t v = predictions;
    tensor_t c = writer.reduction("bv->b", castable_t::max, v);
    // v = v - c
    v = writer.ew("bv,b->bv", scalarop_t::make_sub(dtype), v, c);
    // ev = exp(v)
    tensor_t ev = writer.ew(scalarop_t::make_exp(dtype), v);
    // evsubset{b} = sum_v ev{b,v}*labels{b,v}
    tensor_t evsubset = writer.contraction("bv,bv->b", ev, labels);
    tensor_t evsum    = writer.reduction("bv->b", castable_t::add, ev);

    tensor_t lll = writer.ew(
      "b,b->b",
      scalarop_t::make_div(dtype),
      evsubset, evsum);

    lll = writer.ew(scalarop_t::make_log(dtype), lll);

    // (would like to use unsqueeze here but it is not implemented)

    double one_over_bsz = 1.0 / double(margs.batch_size);
    loss = lll.scale(scalar_t(dtype, write_with_ss(one_over_bsz)));
  }
  loss.save_inplace();

  vector<tensor_t> ws;
  for(auto [name, tensor]: model.weight_map()) {
    if(name.find("lora") != string::npos) {
      ws.push_back(tensor);
    } else {
      if(lora_rank) {
        // we're doing lora, so explicitly save all the weights
        tensor.save_inplace();
      }
    }
  }

  vector<int> checkpoints = vector_from_each_method(
    model.checkpoints, int, get_id);

  map<int, int> forward_labels;
  for(int gid = 0; gid != writer.get_graph().nodes.size(); ++gid) {
    int label = get_label(writer.get_graph(), checkpoints, gid);
    forward_labels.insert({gid, label});
  }

  vector<int> _f = vector_iota<int>(writer.get_graph().nodes.size());
  set<int> forward_ids(_f.begin(), _f.end());

  vector<tensor_t> grads = writer.backprop(loss, ws);

  scalarop_t update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(),
      scalarop_t::make_scale("learning_rate")
    });

  vector<tensor_t> update_ws;
  for(int i = 0; i != grads.size(); ++i) {
    tensor_t& w = ws[i];
    tensor_t& g = grads[i];
    update_ws.push_back(writer.ew("ij,ij->ij", update, w, g));
    update_ws.back().save_inplace();
  }

  return ffinfo_t {
    .graph = writer.get_graph(),
    .x = embeddings.get_id(),
    .y = labels.get_id(),
    .ows = vector_from_each_method(ws, int, get_id),
    .nws = vector_from_each_method(update_ws, int, get_id),
    .loss = loss.get_id(),
    .checkpoints = checkpoints,
    .forward_labels = forward_labels
  };
}

int get_min_checkpoint(
  graph_t const& graph,
  map<int, int> const& forward_labels,
  map<int, int> const& backward_labels,
  int id,
  int ret)
{
  {
    auto iter = forward_labels.find(id);
    if(iter != forward_labels.end()) {
      return std::min(ret, iter->second);
    }
  }
  {
    auto iter = backward_labels.find(id);
    if(iter != backward_labels.end()) {
      return std::min(ret, iter->second);
    }
  }

  auto const& node = graph.nodes[id];
  for(int const& inn_id: node.get_inns_set()) {
    ret = get_min_checkpoint(graph, forward_labels, backward_labels, inn_id, ret);
  }

  return ret;
}

struct graph_id_manager_t {
  struct context_t {
    context_t(graph_id_manager_t* s, int w)
      : self(s), which(w)
    {}

    optional<int> get(int fid) const {
      return self->get(which, fid);
    }

    void insert(int fid, int nid) {
      return self->insert(which, fid, nid);
    }
  private:
    graph_id_manager_t* self;
    int which;
  };

  context_t make_context(int which) {
    return context_t(this, which);
  }

  void insert(int which, int fid, int sid) {
    auto& [fid_to_sid, sid_to_fid] = data[which];
    {
      auto [_, did_insert] = fid_to_sid.insert({fid, sid});
      if(!did_insert) {
        throw std::runtime_error("graph_id_manager did not insert: fid_to_sid");
      }
    }
    {
      auto [_, did_insert] = sid_to_fid.insert({sid, fid});
      if(!did_insert) {
        throw std::runtime_error("graph_id_manager did not insert: sid_to_fid");
      }
    }
  }

  optional<int> get(int which, int fid) const {
    return get_sid(which, fid);
  }

  optional<int> get_sid(int which, int fid) const {
    auto diter = data.find(which);
    if(diter == data.end()) {
      return std::nullopt;
    }
    auto const& m = std::get<0>(diter->second);
    auto iter = m.find(fid);
    if(iter == m.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  optional<int> get_fid(int which, int sid) const {
    auto diter = data.find(which);
    if(diter == data.end()) {
      return std::nullopt;
    }
    auto const& m = std::get<1>(diter->second);
    auto iter = m.find(sid);
    if(iter == m.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  optional<int> get_sid_from_sid(int which_src, int src_sid, int which_dst) const {
    auto maybe = get_fid(which_src, src_sid);
    if(maybe) {
      return get_sid(which_dst, maybe.value());
    } else {
      return std::nullopt;
    }
  }

  void print() const {
    for(auto const& [which, ms]: data) {
      auto const& [fid_to_sid, _] = ms;
      for(auto const& [fid, sid]: fid_to_sid) {
        DOUT("fid " << fid << " | (which " << which << ": " << sid << ")");
      }
    }
  }

private:
  // which -> (fid -> sid, sid -> fid)
  map<int, tuple<map<int, int>, map<int, int>>> data;
};

struct subgraph_former_t
{
  graph_t const& full_graph;
  map<int, int> const& forward_labels;
  map<int, int> const& backward_labels;
  int num_forward_sections;

  bool is_forward_section_input(int fid) const {
    int which = get_forward_section(fid);
    auto const& node = full_graph.nodes[fid];

    if(node.inns.size() == 0) {
      return true;
    }

    if(node.op.is_formation()) {
      int inn_fid = node.inns[0];
      if(get_forward_section(inn_fid) != which) {
        throw std::runtime_error("division across formation edge!");
      }
      return is_forward_section_input(inn_fid);
    }

    for(int const& inn_fid: node.get_inns_set()) {
      if(get_forward_section(inn_fid) != which) {
        return true;
      }
    }

    return false;
  }

  int copy_from_forward_section_inputs(
    graph_id_manager_t::context_t id_context,
    graph_t& ret,
    int fid) const
  {
    {
      auto maybe = id_context.get(fid);
      if(maybe) {
        return maybe.value();
      }
    }

    if(is_forward_section_input(fid)) {
      auto const& op = full_graph.nodes[fid].op;
      int nid = ret.insert_input(op.out_shape(), op.out_dtype());
      id_context.insert(fid, nid);
      return nid;
    }

    auto const& node = full_graph.nodes[fid];
    vector<int> new_inns;
    for(int const& inn_fid: node.inns) {
      int new_inn = copy_from_forward_section_inputs(
        id_context, ret, inn_fid);
      new_inns.push_back(new_inn);
    }

    int nid = ret.insert(node.op, new_inns);
    id_context.insert(fid, nid);
    return nid;
  }

  int get_backward_section_nid(
    int which,
    graph_id_manager_t::context_t id_context,
    graph_t& ret,
    int fid) const
  {
    {
      auto maybe = id_context.get(fid);
      if(maybe) {
        return maybe.value();
      }
    }


    // Assumption: At this point, fid is not on backward section `which`

    auto [is_forward, fid_which] = get_section(fid);
    if(!is_forward && fid_which == which) {
      throw std::runtime_error("this fid should've been in the manager");
    }

    // If this node is
    // 1. from a backward section or
    // 2. from the last forward section,
    // Then add it as an input and depend on it directly

    bool add_as_input = false;
    if(is_forward && fid_which + 1 == num_forward_sections) {
      add_as_input = true;
    }
    if(!is_forward) {
      add_as_input = true;
    }

    if(add_as_input) {
      auto const& op = full_graph.nodes[fid].op;
      int nid = ret.insert_input(op.out_shape(), op.out_dtype());
      id_context.insert(fid, nid);
      return nid;
    }

    // Otherwise, recurse until hitting an input from the
    // dependent forward-section.

    return copy_from_forward_section_inputs(id_context, ret, fid);
  }

  graph_t form(graph_id_manager_t::context_t id_context, int which) const {
    graph_t ret;
    for(int const& fid: get_ordered_section(false, which)) {
      auto const& node = full_graph.nodes[fid];
      vector<int> new_inns;
      for(int const& inn_fid: node.inns) {
        int inn_nid = get_backward_section_nid(
          which, id_context, ret, inn_fid);
        new_inns.push_back(inn_nid);
      }
      int nid = ret.insert(node.op, new_inns);
      id_context.insert(fid, nid);
    }

    return ret;
  }

  graph_t forward(graph_id_manager_t::context_t id_context) const {
    graph_t ret;
    for(int const& fid: full_graph.get_flop_order()) {
      if(forward_labels.count(fid) == 0) {
        continue;
      }
      auto const& node = full_graph.nodes[fid];
      vector<int> new_inns;
      for(int const& inn: node.inns) {
        new_inns.push_back(id_context.get(inn).value());
      }
      int nid = ret.insert(node.op, new_inns);
      id_context.insert(fid, nid);
    }

    return ret;
  }

  tuple<bool, int> get_section(int fid) const {
    {
      auto iter = forward_labels.find(fid);
      if(iter != forward_labels.end()) {
        return { true, iter->second };
      }
    }

    {
      auto iter = backward_labels.find(fid);
      if(iter != backward_labels.end()) {
        return { false, iter->second };
      }
    }

    throw std::runtime_error("should not occur");
  }

  int get_forward_section(int fid) const {
    return forward_labels.at(fid);
  }

  int get_backward_section(int fid) const {
    return backward_labels.at(fid);
  }

  vector<int> get_ordered_section(bool forward, int which) const {
    vector<int> ret;
    for(int const& fid: full_graph.get_flop_order()) {
      auto [is_f, section] = get_section(fid);
      if(section != which) {
        continue;
      }
      if(is_f && forward) {
        ret.push_back(fid);
      } else if(!is_f && !forward) {
        ret.push_back(fid);
      }
    }
    return ret;
  }
};

struct remap_former_t {
  graph_t const& full_graph;

  graph_id_manager_t& id_manager;
  vector<graph_t>& graphs;

  void turn_off_saves() {
    for(auto& graph: graphs) {
      for(int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto& node = graph.nodes[gid];
        node.op.set_save(false);
      }
    }
  }

  // Make sure that fid exists on this graph and is a save
  int set_save(int which, int full_id) {
    {
      auto maybe = id_manager.get_sid(which, full_id);
      if(maybe) {
        int const& sid = maybe.value();
        graphs[which].nodes[sid].op.set_save(true);
        return sid;
      }
    }

    // In this case, this graph does not have the save node so it needs
    // to get the save node from the previous graph.
    if(which == 0) {
      throw std::runtime_error("can't set save, missing node");
    }

    // get the node from the previous graph
    int prev_id = set_save(which - 1, full_id);

    // insert the new node as an input
    auto const& prev_node = graphs[which-1].nodes[prev_id];
    int curr_id = graphs[which].insert_input(
      prev_node.op.out_shape(),
      prev_node.op.out_dtype());

    // and make sure to save it
    graphs[which].nodes[curr_id].op.set_save(true);

    // and make sure to update the id manager
    id_manager.insert(which, full_id, curr_id);

    return curr_id;
  }

  map<int, int> remap_saves_to_full() {
    int which = graphs.size() - 1;
    map<int, int> ret;
    for(int fid = 0; fid != full_graph.nodes.size(); ++fid) {
      auto const& fnode = full_graph.nodes[fid];
      if(fnode.op.is_save()) {
        int sid = set_save(which, fid);
        ret.insert({sid, fid});
      }
    }
    return ret;
  }

  map<int, int> remap_inputs_from_full() const {
    map<int, int> ret;
    graph_t const& g = graphs[0];
    for(int sid = 0; sid != g.nodes.size(); ++sid) {
      auto const& node = g.nodes[sid];
      if(node.op.is_input()) {
        int fid = id_manager.get_fid(0, sid).value();
        ret.insert({fid, sid});
      }
    }
    return ret;
  }

  map<int, int> remap_to(int which_dst) {
    if(which_dst == 0) {
      throw std::runtime_error("invalid which_dst");
    }
    map<int, int> ret;
    graph_t const& dst_g = graphs[which_dst];
    for(int dst_id = 0; dst_id != dst_g.nodes.size(); ++dst_id) {
      auto const& dst_node = dst_g.nodes[dst_id];
      if(dst_node.op.is_input()) {
        int full_id = id_manager.get_fid(which_dst, dst_id).value();
        int src_id = set_save(which_dst - 1, full_id);
        ret.insert({src_id, dst_id});
      }
    }
    return ret;
  }

};

vector<map<int, int>> form_all_remaps(
  graph_t const& full_graph,
  graph_id_manager_t& id_manager,
  vector<graph_t>& graphs)
{
  remap_former_t former {
    .full_graph = full_graph,
    .id_manager = id_manager,
    .graphs = graphs
  };

  former.turn_off_saves();

  vector<map<int, int>> ret;

  // Note: to get the full remaps, it is very important to do
  //       the remaps in reverse, since then all the saves will
  //       be properly added

  ret.push_back(former.remap_saves_to_full());

  for(int i = graphs.size() - 1; i >= 1; --i) {
    ret.push_back(former.remap_to(i));
  }

  ret.push_back(former.remap_inputs_from_full());

  std::reverse(ret.begin(), ret.end());
  return ret;
}

tuple<vector<tuple<uint64_t, uint64_t>>, set<int>, tuple<uint64_t, uint64_t>>
memory_usage(
  graph_t const& graph,
  set<int> keys,
  tuple<uint64_t, uint64_t> flops_usage,
  std::function<vector<string>(int)> make_message)
{
  vector<int> cnts;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    cnts.push_back(node.outs.size());
  }

  vector<tuple<uint64_t, uint64_t>> ret;
  auto& [flops, usage] = flops_usage;
  for(int const& gid: graph.get_flop_order()) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      if(keys.count(gid) == 0) {
        throw std::runtime_error("keys must have all inputs!");
      }
    } else {
      keys.insert(gid);
      usage += graph.out_size(gid);
    }
    if(node.op.is_einsummable()) {
      flops += product(node.op.get_einsummable().join_shape);
    }
    ret.emplace_back(flops, usage);

    for(auto const& msg: make_message(gid)) {
      DOUT(msg << flops << "," << usage)
    }

    for(int inn_gid: node.get_inns_set()) {
      int& cnt = cnts[inn_gid];
      cnt -= 1;
      auto const& inn_node = graph.nodes[inn_gid];
      if(cnt == 0 && !inn_node.op.is_save()) {
        usage -= graph.out_size(inn_gid);
        keys.erase(inn_gid);
      }
    }
  }

  return { ret, keys, flops_usage };
}

template <typename T>
vector<T> make_vec(set<T> const& xs) {
  return vector<T>(xs.begin(), xs.end());
}
template <typename T, typename U>
vector<tuple<T, U>> make_vec(map<T, U> const& xs) {
  return vector<tuple<T, U>>(xs.begin(), xs.end());
}

set<int> remap_keys(map<int, int> const& m, set<int> const& ks)
{
  set<int> ret;
  for(auto const& k: ks) {
    ret.insert(m.at(k));
  }
  return ret;
}

int main(int argc, char** argv) {
  args_t pargs(argc, argv);

  pargs.set_default<uint64_t>("batch",       1   );
  pargs.set_default<uint64_t>("hidden",      4096);
  pargs.set_default<uint64_t>("dim",         4096);

  uint64_t batch       = pargs.get<uint64_t>("batch");
  uint64_t hidden      = pargs.get<uint64_t>("hidden");
  uint64_t dim         = pargs.get<uint64_t>("dim");

  pargs.set_default<int>("nrep", 10);
  int nrep = pargs.get<int>("nrep");

  pargs.set_default<int>("num_ws", 3);
  int num_ws = pargs.get<int>("num_ws");

  ///////////////////////////////////////////////////////////////////

  pargs.set_default<uint64_t>("seqlen", 2048);
  pargs.set_default<int>("n_layers", -1);

  uint64_t seqlen = pargs.get<uint64_t>("seqlen");
  int n_layers = pargs.get<int>("n_layers");

  ///////////////////////////////////////////////////////////////////

  pargs.set_default<bool>("llama", true);
  bool do_llama = pargs.get<bool>("llama");

  auto info = do_llama                         ?
    make_llama_7B(batch, seqlen, n_layers)     :
    make_ff_simple(batch, hidden, dim, num_ws) ;

  vector<string> cs {
    "#ff5733", "#377e20", "#2cb392", "#7af3f5", "#374ded", "#b337ed"
  };

  map<int, string> colors;
  //for(int id: info.checkpoints) {
  //  colors.insert({id, "yellow"});
  //}
  map<int, int> backward_labels;
  int max_backward_label = -1;
  for(int id = 0; id != info.graph.nodes.size(); ++id) {
    auto iter = info.forward_labels.find(id);
    if(iter != info.forward_labels.end()) {
      int cp = iter->second;
      colors.insert({id, cs.at(cp % cs.size())});
      //colors.insert({id, "gray"});
    } else {
      int cpoint = get_min_checkpoint(
        info.graph,
        info.forward_labels, backward_labels,
        id,
        info.checkpoints.size()
      );
      backward_labels.insert({id, cpoint});
      max_backward_label = std::max(max_backward_label, cpoint);

      colors.insert({id, cs.at((cpoint+1) % cs.size())});
    }
  }

  {
    string filename = "g.gv";
    std::ofstream f(filename);
    info.graph.print_graphviz(f, colors);
    DOUT("printed " << filename);
  }

  ///////////////////////////////////////////////////////////////////
  subgraph_former_t former {
    .full_graph = info.graph,
    .forward_labels = info.forward_labels,
    .backward_labels = backward_labels,
    .num_forward_sections = int(info.checkpoints.size())
  };

  graph_id_manager_t manager;
  vector<graph_t> graphs;

  graphs.push_back(former.forward(manager.make_context(0)));

  {
    string filename = "gforward.gv";
    std::ofstream f(filename);
    graphs.back().print_graphviz(f);
    DOUT("printed " << filename);
  }

  for(int i = max_backward_label; i >= 0; --i) {
    int which = graphs.size();
    graphs.push_back(former.form(manager.make_context(which), i));

    {
      string filename = "g" + write_with_ss(i) + ".gv";
      std::ofstream f(filename);
      graphs.back().print_graphviz(f);
      DOUT("printed " << filename);
    }
  }

  vector<map<int, int>> remaps = form_all_remaps(info.graph, manager, graphs);

  for(int i = 0; i != graphs.size(); ++i) {
    string filename = "f" + write_with_ss(i) + ".gv";
    std::ofstream f(filename);
    graphs[i].print_graphviz(f);
    DOUT("printed " << filename);
  }

  tuple<uint64_t, uint64_t> flops_usage(0, 0);
  auto& [flops, usage] = flops_usage;
  set<int> keys;
  for(int gid = 0; gid != info.graph.nodes.size(); ++gid) {
    auto const& node = info.graph.nodes[gid];
    if(node.op.is_input()) {
      usage += info.graph.out_size(gid);
      keys.insert(gid);
    }
  }

  auto make_base_message = [&](int gid) {
    vector<string> ret;
    auto const& node = info.graph.nodes[gid];
    for(auto const& inn: node.get_inns_set()) {
      for(auto const& w: info.ows) {
        if(w == inn) {
          ret.push_back("weight:" + write_with_ss(inn) + ",");
        }
      }
    }
    return ret;
  };

  int _which;
  auto _make_subgraph_message = [&](int sid)
  {
    vector<string> ret;
    int gid = manager.get_fid(_which, sid).value();
    auto const& node = info.graph.nodes[gid];
    for(auto const& inn: node.get_inns_set()) {
      for(auto const& w: info.ows) {
        if(w == inn) {
          ret.push_back("weight:" + write_with_ss(inn) + ",");
        }
      }
    }
    return ret;
  };
  auto make_subgraph_message = [&](int which) {
    _which = which;
    return _make_subgraph_message;
  };

  tuple<uint64_t, uint64_t> base_flops_usage(flops_usage);
  vector<tuple<uint64_t, uint64_t>> base_usages;
  set<int> base_keys = keys;
  {
    auto [us, ks, u] = memory_usage(
      info.graph, base_keys, base_flops_usage, make_base_message);
    base_usages = us;
    base_keys = ks;
    base_flops_usage = u;
  }

  vector<tuple<uint64_t, uint64_t>> usages;
  for(int i = 0; i != graphs.size(); ++i) {
    map<int, int> const& remap = remaps[i];
    keys = remap_keys(remap, keys);
    graph_t const& graph = graphs[i];
    auto [us, ks, u] = memory_usage(
      graph, keys, flops_usage, make_subgraph_message(i));
    vector_concatenate_into(usages, us);
    keys = ks;
    flops_usage = u;
    DOUT("v:" << flops);
  }

  keys = remap_keys(remaps.back(), keys);

  if(!vector_equal(make_vec(keys), make_vec(base_keys))) {
    throw std::runtime_error("!");
  }

  for(auto const& [f,u]: base_usages) {
    DOUT("base__:" << f << "," << u);
  }
  for(auto const& [f,u]: usages) {
    DOUT("cpoint:" << f << "," << u);
  }
}
