#include "../base/setup.h"

namespace build_trees_ns {

template <typename T>
struct bgraph_t {
  struct node_t {
    set<int> outs;
    set<int> inns;
    T item;

    vector<int> inns_as_vec() const {
      return vector<int>(inns.begin(), inns.end());
    }
    vector<int> outs_as_vec() const {
      return vector<int>(outs.begin(), outs.end());
    }
  };

  map<int, node_t> const& get() const { return items; }

  set<int> const& get_inns(int id) const {
    return items.at(id).inns;
  }
  set<int> const& get_outs(int id) const {
    return items.at(id).outs;
  }

  void insert(int id, T const& value, set<int> const& inns, set<int> const& outs)
  {
    for(int const& inn: inns) {
      if(!has_id(inn)) {
        throw std::runtime_error("does not have this input");
      }
    }
    for(int const& out: inns) {
      if(!has_id(out)) {
        throw std::runtime_error("does not have this output");
      }
    }
    auto [_, did_insert] = items.insert(
      {id, node_t { .outs = outs, .inns = inns, .item = value }});

    for(auto const& inn: inns) {
      get_node(inn).outs.insert(id);
    }
    for(auto const& out: outs) {
      get_node(out).inns.insert(id);
    }
  }

  void insert(int id, T const& value, set<int> const& inns) {
    set<int> outs{};
    return insert(id, value, inns, outs);
  }

  void insert_edge(int inn, int out) {
    auto& inn_node = get_node(inn);
    auto& out_node = get_node(out);
    inn_node.outs.insert(out);
    out_node.inns.insert(inn);
  }

  void remove(int id) {
    auto iter = items.find(id);
    if(iter == items.end()) {
      throw std::runtime_error("remove does not have id");
    }
    node_t const& node = iter->second;

    for(auto const& inn: node.inns) {
      if(node.outs.erase(inn) == 0) {
        throw std::runtime_error("could not erase inn's out.");
      }
    }

    for(auto const& out: node.outs) {
      if(node.inns.erase(out) == 0) {
        throw std::runtime_error("could not erase out's inn.");
      }
    }
  }

  vector<int> dfs_to_inns_from(int id) const {
    vector<int> to_process{id};
    vector<int> ret;
    while(to_process.size() != 0) {
      ret.push_back(id);
      to_process.pop_back();
      auto const& node = get_node(id);
      vector_concatenate_into(to_process, node.inns_as_vec());
    }
    return ret;
  }

  vector<int> dfs_to_outs_from(int id) const {
    vector<int> to_process{id};
    vector<int> ret;
    while(to_process.size() != 0) {
      ret.push_back(id);
      to_process.pop_back();
      auto const& node = get_node(id);
      vector_concatenate_into(to_process, node.outs_as_vec());
    }
    return ret;
  }

  T const& operator[](int key) const {
    return get_node(key).item;
  }

  T& operator[](int key) {
    return get_node(key).item;
  }

  int get_max_id() const {
    return items.size() == 0 ? -1 : items.rbegin()->first;
  }

  bool has_id(int key) const {
    return items.count(key) > 0;
  }

  template <typename U>
  bgraph_t<U> map_items_with_id(std::function<U(int,T const&)> f) const {
    map<int, typename bgraph_t<U>::node_t> new_items;
    for(auto const& [id, node]: items) {
      new_items.insert({
        id,
        typename bgraph_t<U>::node_t {
          .outs = node.outs,
          .inns = node.inns,
          .item = f(id, node.item)
        }
      });
    }

    ///
    bgraph_t<U> ret;
    ret.items = std::move(new_items);
    return ret;
  }

  template <typename U>
  bgraph_t<U> map_items(std::function<U(T const&)> f) const {
    return map_items_with_id([&f](int id, T const& t) {
      return f(t);
    });
  }

  struct is_connected_t {
    is_connected_t(map<int, set<int>> && v)
      : all_inns_of(v)
    {}

    bool operator()(int src, int dst) const {
      return is_path_from_to(src, dst);
    }
    bool is_path_from_to(int src, int dst) const {
      if(src == dst) {
        return true; // the trivial path
      }
      return all_inns_of.at(dst).count(src) > 0;
    }

  private:
    map<int, set<int>> all_inns_of;
  };

  is_connected_t collect_is_path_from_to() const {
    map<int, set<int>> ret;
    for(auto const& id: dag_order_inns_to_outs()) {
      if(ret.count(id) > 0) {
        continue;
      }

      auto const& node = items.at(id);

      set<int> all_children = node.inns;
      for(auto const& inn: node.inns) {
        auto const& s = ret.at(inn);
        all_children.insert(s.begin(), s.end());
      }

      ret.insert({id, std::move(all_children)});
    }

    return is_connected_t(std::move(ret));
  }

  vector<int> dag_order_inns_to_outs() const {
    map<int, int> counts;
    vector<int> ret;
    ret.reserve(items.size());
    for(auto const& [id, node]: items) {
      int num_inn = node.inns.size();
      if(num_inn == 0) {
        ret.push_back(id);
      } else {
        counts.insert({id, num_inn});
      }
    }

    int idx = 0;
    int stop = ret.size();
    while(idx != ret.size()) {
      for(; idx != stop; ++idx) {
        int id = ret[idx];
        auto const& node = items.at(id);
        for(auto const& out: node.outs) {
          auto iter = counts.find(out);
          if(iter == counts.end()) {
            throw std::runtime_error("should not happen");
          }
          int& cnt = iter->second;
          cnt--;
          if(cnt == 0) {
            ret.push_back(out);
            counts.erase(iter);
          }
        }
      }
      stop = ret.size();
    }

    return ret;
  }

  vector<int> dag_order_outs_to_inns() const {
    map<int, int> counts;
    vector<int> ret;
    ret.reserve(items.size());
    for(auto const& [id, node]: items) {
      int num_out = node.outs.size();
      if(num_out == 0) {
        ret.push_back(id);
      } else {
        counts.insert({id, num_out});
      }
    }

    int idx = 0;
    int stop = ret.size();
    while(idx != ret.size()) {
      for(; idx != stop; ++idx) {
        int id = ret[idx];
        auto const& node = items.at(id);
        for(auto const& inn: node.inns) {
          auto iter = counts.find(inn);
          if(iter == counts.end()) {
            throw std::runtime_error("should not happen");
          }
          int& cnt = iter->second;
          cnt--;
          if(cnt == 0) {
            ret.push_back(inn);
            counts.erase(iter);
          }
        }
      }
      stop = ret.size();
    }

    return ret;
  }

  void print_graphviz(std::ostream& out, map<int, string> get_color) const {

    using std::endl;
    string tab = "  ";
    out << "digraph {" << endl;
    for(auto const& id: dag_order_inns_to_outs()) {
      auto const& node = items.at(id);
      string label = write_with_ss(id);
      auto iter = get_color.find(id);
      string color = "gray";
      if(iter != get_color.end()) {
        color = iter->second;
      }
      out << tab
          << "n" << id
          << " [style=filled,label=\"" << label << "\""
          << ",color=\"" << color << "\""
          << "]" << endl;
      for(auto const& inn: node.inns) {
        out << tab << "n" << inn << " -> " << "n" << id << endl;
      }
    }
    out << "}" << endl;
  }
  void print_graphviz(std::ostream& out) const {
    map<int, string> get_color;
    print_graphviz(out, get_color);
  }

private:
  map<int, node_t> items;

private:
  node_t const& get_node(int key) const {
    return items.at(key);
  }

  node_t& get_node(int key) {
    return items.at(key);
  }
};

template <typename T>
class btree_t {
  struct node_t {
    optional<int> out;
    set<int> inns;
    T item;

    bool is_root() const { return !bool(out); }
    bool is_leaf() const { return inns.size() == 0; }

    vector<int> inns_as_vec() const {
      return vector<int>(inns.begin(), inns.end());
    }
  };

  optional<int> root_id;
  map<int, node_t> items;

public:
  void insert_root(int id, T const& value) {
    if(root_id) {
      throw std::runtime_error("tree already has root");
    }
    root_id = id;
    auto [_, did_insert] = items.insert({
      id,
      node_t { .out = std::nullopt, .inns = {}, .item = value }
    });
    if(!did_insert) {
      throw std::runtime_error("already have this id: insert_root");
    }
  }

  void insert_leaf(int id, int out, T const& value) {
    auto& out_node = get_node(out);

    auto [_, did_insert] = items.insert({
      id,
      node_t { .out = out, .inns = {}, .item = value }
    });
    if(!did_insert) {
      throw std::runtime_error("already have this id: insert_leaf");
    }

    out_node.inns.insert(id);
  }

  void erase_leaf(int id) {
    auto iter = items.find(id);
    if(iter == items.end()) {
      throw std::runtime_error("erase_leaf: does not have id");
    }

    optional<int> const& maybe_out_id = iter->second.out;
    if(maybe_out_id) {
      int const& out_id = maybe_out_id.value();
      auto& out_node = get_node(out_id);
      if(out_node.inns.erase(id) == 0) {
        throw std::runtime_error("this node was not connected to its out");
      }
    }

    items.erase(iter);
  }

  vector<int> dfs_from(int start_id) const {
    vector<int> to_process{start_id};
    vector<int> ret;
    while(to_process.size() != 0) {
      int id = to_process.back();
      to_process.pop_back();

      ret.push_back(id);
      auto const& node = get_node(id);
      vector_concatenate_into(to_process, node.inns_as_vec());
    }
    return ret;
  }

  vector<int> dfs_from_root() const {
    if(root_id) {
      return dfs_from(root_id.value());
    } else {
      if(items.size() != 0) {
        throw std::runtime_error("root_id is nullopt, must have empty items");
      }
      return vector<int>();
    }
  }

  T const& operator[](int key) const {
    return get_node(key).item;
  }

  T& operator[](int key) {
    return get_node(key).item;
  }

  bgraph_t<T> as_graph() const {
    bgraph_t<T> graph;
    vector<int> ids = dfs_from_root();
    while(ids.size() != 0) {
      int id = ids.back();
      ids.pop_back();

      auto const& node = get_node(id);
      graph.insert(id, node.item, node.inns);
    }
    return graph;
  }

  bool is_singleton() const {
    if(root_id) {
      return get_node(root_id.value()).is_leaf();
    } else {
      return false;
    }
  }

  bool is_leaf(int id) const {
    return get_node(id).is_leaf();
  }

  bool is_empty() const {
    return !bool(root_id);
  }

  int get_root_id() const {
    return root_id.value();
  }

  set<int> const& get_inns(int id) const {
    return get_node(id).inns;
  }
  vector<int> get_inns_as_vec(int id) const {
    return get_node(id).inns_as_vec();
  }
private:
  node_t const& get_node(int key) const {
    return items.at(key);
  }

  node_t& get_node(int key) {
    return items.at(key);
  }
};

template <typename T, typename F>
bool _tree_can_consume(
  bgraph_t<T> const& graph,
  F& is_path_from_to,
  btree_t<T> const& tree,
  int consume,
  int dst)
{
  // Assumption: consume -> dst is an edge in graph and dst is in the tree.

  // Return whether or not there is no path from consume to any node in the tree
  // besides (consume->dst).
  //
  // The reasoning is that a node can't be consumed if doing so would make some
  // of it's outgoing edges impossible to fill.

  // Consider 5    -> 7
  //          v       ^
  //           --> 6 -^
  // Where tree = {7} -- i.e. the singleton tree.
  //
  // * 5 into {7} fails because {5->7} < {6} and {5->7} > {6} because of
  //   edge 5->6 (a graph of all trees is no longer a dag)
  // * 5 into {6->7} at 6  fails  because edge 5->7 would be lost
  // * 5 into {6->7} at 7  fails  because edge 5->6 would be lost
  // * 6 into {7} is fine

  auto tree_ids = tree.dfs_from_root();
  for(auto consume_out: graph.get_outs(consume)) {
    if(consume_out != dst) {
      for(auto const& tree_id: tree_ids) {
        if(is_path_from_to(consume_out, tree_id)) {
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
set<int> _compute_tree_outs(
  bgraph_t<T> const& graph,
  btree_t<T> const& tree,
  map<int, int> const& id_to_root)
{
  set<int> tree_ids;
  {
    auto ids = tree.dfs_from_root();
    tree_ids = set<int>(ids.begin(), ids.end());
  }

  set<int> ret;
  for(auto const& id: tree_ids) {
    set<int> const& outs = graph.get_outs(id);
    for(auto const& out: outs) {
      if(tree_ids.count(out) == 0) {
        ret.insert(id_to_root.at(out));
      }
    }
  }
  return ret;
}

template <typename T>
set<int> _get_tree_outs(
  bgraph_t<T> const& graph,
  btree_t<T> const& tree,
  map<int, int> const& id_to_root)
{
  if(tree.is_empty()) {
    return {};
  }

  int root_id = tree.get_root_id();
  set<int> outs;
  for(int const& id: tree.dfs_from_root()) {
    for(auto const& out: graph.get_outs(id)) {
      auto out_root_id = id_to_root.at(out);
      if(out_root_id != root_id) {
        outs.insert(out_root_id);
      }
    }
  }
  return outs;
}

template <typename T>
bgraph_t<btree_t<T>> build_trees(bgraph_t<T> const& graph)
{
  auto is_path_from_to = graph.collect_is_path_from_to();

  auto const& gitems = graph.get();

  bgraph_t<btree_t<T>> ret;

  map<int, int> id_to_root;

  for(auto const& root_id: graph.dag_order_outs_to_inns()) {
    if(id_to_root.count(root_id) > 0) {
      continue;
    }

    ret.insert(root_id, btree_t<T>(), {});
    auto& tree = ret[root_id];
    tree.insert_root(root_id, graph[root_id]);
    id_to_root.insert({root_id, root_id});

    vector<int> to_process{root_id};
    int idx = 0;
    for(; idx != to_process.size(); ++idx) {
      int id = to_process[idx];
      for(int const& inn_id: graph.get_inns(id)) {
        if(id_to_root.count(inn_id) > 0) {
          continue;
        }
        bool can_consume = _tree_can_consume(
          graph, is_path_from_to, tree, inn_id, id);
        if(can_consume) {
          tree.insert_leaf(inn_id, id, graph[inn_id]);
          to_process.push_back(inn_id);
          id_to_root.insert({inn_id, root_id});
        } else {
          // could not consume inn_id
        }
      }
    }
  }

  for(auto const& [root_id, node]: ret.get()) {
    auto const& tree = node.item;
    for(int const& out_root_id: _get_tree_outs(graph, tree, id_to_root)) {
      ret.insert_edge(root_id, out_root_id);
    }
  }

  return ret;
}

}
