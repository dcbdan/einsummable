#include "super.h"

int super_graph_t::insert(
  memgraph_t const& memgraph,
  vector<int> op_mids)
{
  node_t ret {
    .ops = op_mids,
    .inns = set<int>(),
    .outs = set<int>()
  };

  // For each op, add the dependent super ops, if any
  set<int> here(op_mids.begin(), op_mids.end());
  for(int op_mid: op_mids) {
    auto const& node = memgraph.nodes[op_mid];
    for(int const& inn_mid: node.inns) {
      auto const& inn_node = memgraph.nodes[inn_mid];

      // inputmem and inputstos are satisfied at startup, so don't
      // include them
      if(inn_node.op.is_inputmem() || inn_node.op.is_inputsto()) {
        continue;
      } 

      if(here.count(inn_mid) == 0) {
        // there must be a dependent super node
        auto iter = mid_to_sid.find(inn_mid);
        if(iter == mid_to_sid.end()) {
          throw std::runtime_error(
            "external input mid "+write_with_ss(inn_mid)+": no corresponding sid");
        }

        ret.inns.insert(iter->second);
      } else {
        // this input will be done in this super node
      }
    }
  }

  nodes.push_back(ret);

  int sid = nodes.size() - 1;
  for(int const& inn: ret.inns) {
    nodes[inn].outs.insert(sid);
  };

  for(auto const& op_mid: op_mids) {
    mid_to_sid.insert({op_mid, sid});
  }

  return sid;
}

void super_graph_t::print_graphviz(std::ostream& out) {
  using std::endl;
  string tab = "  ";
  out << "digraph {" << endl;
  for(int id = 0; id != nodes.size(); ++id) {
    string label = "N" + write_with_ss(nodes[id].ops.size());
    auto const& node = nodes[id];
    out << tab
        << "n" << id
        << " [style=filled,label=\"" << label << "\""
        << "]" << endl;
    for(int const& inn_id: node.inns) {
      out << tab << "n" << inn_id << " -> "
          << "n" << id << endl;
    }
  }
  out << "}" << endl;
}

super_graph_t
create_super_graph(memgraph_t const& mg)
{
  super_graph_t ret;

  int nlocs = mg.mem_sizes().size();
  vector<vector<int>> pending(nlocs);

  auto complete = [&](int loc) {
    auto& ps = pending[loc];
    if(ps.size() > 0) {
      ret.insert(mg, ps);
      ps.resize(0);
    }
  };

  auto is_contraction = [&](int mid) {
    auto const& op = mg.nodes[mid].op;
    return op.is_einsummable() && op.get_einsummable().is_contraction();
  };
  auto num_contraction = [&](int loc) {
    int ret = 0;
    for(int const& mid: pending[loc]) {
      if(is_contraction(mid)) {
        ret++;
      }
    }
    return ret;
  };
  auto add_to_pending_max_contraction = [&](int loc, int mid) {
    if(is_contraction(mid) && num_contraction(loc) >= 1) { // TODO just hardcoding for now
      complete(loc);
    }
    pending[loc].push_back(mid);
  };

  auto is_move = [&](int mid) {
    return mg.nodes[mid].op.is_move();
  };
  auto add_to_pending_one_move = [&](int loc, int mid) {
    pending[loc].push_back(mid);
    if(is_move(mid)) {
      complete(loc); 
    }
  };

  auto add_to_pending_max_nodes = [&](int loc, int mid) {
    pending[loc].push_back(mid);
    if(pending[loc].size() >= 20) { // TODO just hardcoding for now
      complete(loc);
    }
  };

  auto add_to_pending = add_to_pending_max_nodes;

  int num_inn_nodes = 0;
  for(int const& mid: mg.get_order()) {
    auto const& node = mg.nodes[mid];

    // Don't add input nodes to the super graph
    if(node.op.is_inputsto() || node.op.is_inputmem()) {
      num_inn_nodes++;
      continue;
    }

    int loc = 
      node.op.is_move()                ?
      node.op.get_move().get_src_loc() :
      node.op.get_loc()                ;

    // The invariant is that all dependent ops must be
    // part of the super graph already. This makes sure that 
    // when this node gets added to the super graph, the super
    // dependencies can be accounted for.

    // 1. acquire the other locations to depend on
    set<int> inn_loc_deps;
    for(int const& inn_mid: node.inns) {
      auto const& inn_node = mg.nodes[inn_mid];
      if(inn_node.op.is_move()) {
        inn_loc_deps.insert(inn_node.op.get_move().get_src_loc());
      } else {
        int inn_loc = inn_node.op.get_loc();
        if(inn_loc != loc) {
          inn_loc_deps.insert(inn_loc);
        }
      }
    }

    if(inn_loc_deps.count(loc) != 0) {
      throw std::runtime_error("this should not occur");
    }

    // 2. make sure that the recv locs are completed
    for(int const& inn_loc: inn_loc_deps) {
      complete(inn_loc);
    } 

    // 3. add this node to pending
    add_to_pending(loc, mid);
  }

  for(int loc = 0; loc != nlocs; ++loc) {
    complete(loc);
  }

  if(ret.mid_to_sid.size() + num_inn_nodes != mg.nodes.size()) {
    throw std::runtime_error("incorrect number of mids added");
  }

  for(int i = 0; i != mg.nodes.size(); ++i) {
    auto const& node = mg.nodes[i];
    if(node.op.is_inputmem() || node.op.is_inputsto()) {
      continue;
    }
    if(ret.mid_to_sid.count(i) != 1) {
      throw std::runtime_error("mid not accounted for");
    }
  }

  for(auto const& node: ret.nodes) {
    int con = 0;
    int ein = 0;
    int tou = 0;
    int mov = 0;
    for(auto const& mid: node.ops) {
      auto const& op = mg.nodes[mid].op;
      if(op.is_touch()) {
        tou++;
      } else if(op.is_einsummable()) {
        ein++;
        if(op.get_einsummable().is_contraction()) {
          con++;
        }
      } else if(op.is_move()) {
        mov++;
      }
    }
    int tot = con + ein + tou + mov;
    if(tot > 0) {
      DOUT("con/ein/tou/mov: " << con << "/" << ein << "/" << tou << "/" << mov << "    " <<
           tot);
    }
  }
  

  return ret;
}
