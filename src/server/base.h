#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"
#include "../einsummable/taskgraph.h"
#include "../einsummable/memgraph.h"
#include "../einsummable/relation.h"
#include "../einsummable/dbuffer.h"

#include "../engine/communicator.h"
#include "../engine/threadpool.h"

// Some definitions:
// * compute-node: The place where an excutable is running on. A single
//   server_mg_base_t object will exist on each compute-node. The compute-node
//   can be identified by the communicator's rank.
// * location: The place where a tid lives. A compute-node may have many
//   locations in it's purview. For example: 1 compute-node with 4 gpus (locations).
// * storage-location: The place where data can be offloaded to. Just like a compute-node,
//   it may may have many locations in it's purview. (only relevant with memgraphs)
//
// Assumption:
// * storage-location == compute-node. That is, all locations on a compute-node use
//   the same storage-location.

// The server maintains the following maps:
//   gid -> relation
//      * given a gid, return the relation
//      * this map only needs to be managed at rank zero
//   tid -> data
//      * given a tid, we can acquire a copy of the actual
//        bytes contained therein
// It is up to the user if every tid is part of a gid or if
// the gid map is never used.
struct server_base_t {
  virtual void listen() = 0;

  // Note: this should only be called on ranks not equal to zero.
  virtual void register_listen(string key, std::function<void()> f) = 0;

  // Should only be called by rank zero and when all other
  // ranks are listening {{{
  void insert_gid_without_data(int gid, relation_t const& relation);

protected:
  // This is protected because it will invalidate gid_map.
  virtual void execute(
    taskgraph_t const& taskgraph,
    map<string, scalar_t> const& scalar_vars) = 0;
public:
  // Execute this taskgraph and rewrite the new gid_map... Must be careful
  void execute(
    taskgraph_t const& taskgraph,
    map<int, relation_t> const& new_gid_map,
    map<string, scalar_t> const& scalar_vars = {});

  // create a taskgraph and execute the graph.
  void execute_graph(
    graph_t const& graph,
    vector<placement_t> const& placements,
    map<string, scalar_t> const& scalar_vars = {});

  // Get a relation broadcast across the cluster and put it
  // onto node zero.
  virtual dbuffer_t get_tensor(relation_t const& src_relation) = 0;

  dbuffer_t get_tensor_from_gid(int gid);

  relation_t const& get_relation(int gid) const;

  void insert_constant(
    int gid,
    relation_t const& relation,
    scalar_t value);

  // Get a tensor here and partition it across the cluster
  // into managed data
  void insert_tensor(
    int gid,
    relation_t const& dst_relation,
    dbuffer_t src_tensor);
  void insert_tensor(
    int gid,
    placement_t const& placement,
    dbuffer_t src_tensor);
  void insert_tensor(
    int gid,
    vector<uint64_t> const& shape,
    dbuffer_t src_tensor);

  virtual void insert_constant_relation(
    relation_t const& dst_relation,
    scalar_t value) = 0;
  virtual void insert_relation(
    relation_t const& dst_relation,
    dbuffer_t src_tensor) = 0;

  // repartitions the relations stored in managed data
  // Note: anything not in the dst remap may get
  //       deleted!
  virtual void remap(remap_relations_t const& remap) = 0;

  void remap(map<int, relation_t> const& gid_to_new_relation);

  // Note: gid_map will only contain the dst tids after this
  void remap_gids(vector<tuple<int,int>> const& remap);

  // Get the max tid across all data objects on all ranks.
  // Useful for creating new relations that won't overwrite
  // existing data
  virtual int get_max_tid() = 0;

  virtual void shutdown() = 0;
  // }}}

  virtual string get_registered_cmd() const = 0;

  // it is error if any of the tids in local_insert_tensors and local_erase_tensors
  // do not live on _this_ compute-node.

  // data is a mapping from tid -> (location, buffer) pair
  // (since every compute-node may have multiple locations, we can't tell
  //  where to insert the tensors unless told)
  virtual void local_insert_tensors(
    map<int, tuple<int, buffer_t>> data) = 0;

  virtual void local_erase_tensors(vector<int> const& tids) = 0;

  virtual bool make_parallel_partialize_groups() const = 0;

  // This is used for the repartition funciton, which will create a threadpool
  // if none is given
  virtual threadpool_t* get_cpu_threadpool() { return nullptr; }

private:
  // Note: gid_map only exists at the server
  map<int, relation_t> gid_map;
};

struct server_dist_base_t : server_base_t {
  server_dist_base_t(
    communicator_t& c)
    : comm(c)
  {}

  void listen();

  void register_listen(string key, std::function<void()> f);

  // {{{
protected:
  // This is protected because it will invalidate gid_map.
  void execute(
    taskgraph_t const& taskgraph,
    map<string, scalar_t> const& scalar_vars);

public:
  dbuffer_t get_tensor(relation_t const& src_relation);

  void insert_constant_relation(
    relation_t const& dst_relation,
    scalar_t value);
  void insert_relation(
    relation_t const& dst_relation,
    dbuffer_t src_tensor);
  void insert_relation_helper(
    remap_relations_t remap,
    map<int, buffer_t> data);

  void remap(remap_relations_t const& remap);

  int get_max_tid();

  void shutdown();
  // }}}

  string get_registered_cmd() const {
    return write_with_ss(cmd_t::registered_cmd);
  }

private:
  void convert_remap_to_compute_node(remap_relations_t& remap);

protected:
  communicator_t& comm;

private:
  map<string, std::function<void()>> listeners;

  enum class cmd_t {
    execute_tg = 0,
    remap,
    get_tensor,
    insert_constant,
    insert_relation,
    max_tid,
    registered_cmd,
    shutdown
  };

  static vector<string> const& cmd_strs() {
    static vector<string> ret {
      "execute_tg",
      "remap",
      "get_tensor",
      "insert_constant",
      "insert_relation",
      "max_tid",
      "registered_cmd",
      "shutdown"
    };
    return ret;
  }

  friend std::ostream& operator<<(std::ostream& out, cmd_t const& c);
  friend std::istream& operator>>(std::istream& inn, cmd_t& c);

  // only recvs from rank 0
  cmd_t recv_cmd() {
    return parse_with_ss<cmd_t>(comm.recv_string(0));
  }

  // only from rank zero
  void broadcast_cmd(cmd_t const& cmd) {
    comm.broadcast_string(write_with_ss(cmd));
  }

protected:
  virtual void execute_tg_server(
    taskgraph_t const& taskgraph,
    map<string, scalar_t> const& sclar_vars) = 0;
  virtual void execute_tg_client() = 0;

  void _local_insert_constant_relation(
    relation_t const& dst_relation,
    scalar_t value);

  virtual void remap_server(remap_relations_t const& remap_relations) = 0;
  virtual void remap_client() = 0;

  virtual int local_get_max_tid() const = 0;

  // return a location that exists at this compute-node
  virtual int local_candidate_location() const = 0;

  bool is_local_location(int loc) {
    return comm.get_this_rank() == loc_to_compute_node(loc);
  }
  virtual int loc_to_compute_node(int loc) const = 0;

  virtual buffer_t local_copy_data(int tid) = 0;

  // Note: this remap is with respect to locations
  map<int, buffer_t> local_copy_source_data(remap_relations_t const& remap);
};

std::ostream& operator<<(std::ostream& out, server_dist_base_t::cmd_t const& c);
std::istream& operator>>(std::istream& inn, server_dist_base_t::cmd_t& c);

// This is a server_base object that executes all taskgraphs by first
// converting them into memgraphs.
struct server_mg_base_t : server_dist_base_t {
  server_mg_base_t(
    communicator_t& c,
    allocator_settings_t s = allocator_settings_t::default_settings())
    : server_dist_base_t(c), alloc_settings(s),
      make_parallel_partialize_groups_(false),
      use_storage_(true),
      split_off_inputs_(true)
  {}

  void set_parallel_partialize(bool new_val) {
    this->make_parallel_partialize_groups_ = new_val;
  }

  void set_use_storage(bool new_val) {
    this->use_storage_ = new_val;
  }

  void set_split_off_inputs(bool new_val) {
    this->split_off_inputs_ = new_val;
  }

  // this must be called from all locations
  virtual void execute_memgraph(
    memgraph_t const& memgraph,
    bool for_remap,
    map<string, scalar_t> const& scalar_vars = {}) = 0;
  // the for_remap should be true if the computation is just a remap instead of
  // a general taskgraph

  // server, client pairs {{{
  void execute_tg_server(
    taskgraph_t const& taskgraph,
    map<string, scalar_t> const& scalar_vars);
  void execute_tg_client();

  void remap_server(remap_relations_t const& remap_relations);
  void remap_client();

  struct make_mg_info_t {
    vector<uint64_t> mem_sizes;
    map<int, memstoloc_t> data_locs;
    vector<int> which_storage;
  };
  virtual make_mg_info_t recv_make_mg_info() = 0;
  virtual void           send_make_mg_info() = 0;

  virtual void storage_remap_server(
    vector<vector<std::array<int, 2>>> const& remaps) = 0;
  virtual void storage_remap_client() = 0;

  virtual void rewrite_data_locs_server(map<int, memstoloc_t> const& out_tg_to_loc) = 0;
  virtual void rewrite_data_locs_client() = 0;
  // }}}

  static
  vector<vector<std::array<int, 2>>>
  create_storage_remaps(
    int world_size,
    map<int, memstoloc_t> const& full_data_locs,
    map<int, memstoloc_t> const& inn_tg_to_loc);

  bool make_parallel_partialize_groups() const {
    return make_parallel_partialize_groups_;
  }

public:
  allocator_settings_t alloc_settings;

  bool make_parallel_partialize_groups_;
  bool use_storage_;
  bool split_off_inputs_;
};

map<string, scalar_t> scalar_vars_from_wire(string const& s);
string scalar_vars_to_wire(map<string, scalar_t> const& vars);

