#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"
#include "../einsummable/taskgraph.h"
#include "../einsummable/memgraph.h"
#include "../einsummable/relation.h"
#include "../einsummable/dbuffer.h"

#include "../engine/communicator.h"

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
  virtual void register_listen(string key, std::function<void(server_base_t*)> f) = 0;

  // Should only be called by rank zero and when all other
  // ranks are listening {{{
  virtual void execute(taskgraph_t const& taskgraph) = 0;

  // create a taskgraph and execute the graph.
  void execute_graph(
    graph_t const& graph,
    vector<placement_t> const& placements);

  // Get a relation broadcast across the cluster and put it
  // onto node zero.
  virtual dbuffer_t get_tensor(relation_t const& src_relation) = 0;

  dbuffer_t get_tensor_from_gid(int gid);

  relation_t const& get_relation(int gid) const;

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

  // data is a mapping from tid -> (buffer, location) pair
  // (since every compute-node may have multiple locations, we can't tell
  //  where to insert the tensors unless told)
  virtual void local_insert_tensors(
    map<int, tuple<int, buffer_t>> data) = 0;

  virtual void local_erase_tensors(vector<int> const& tids) = 0;

  virtual bool make_parallel_partialize_groups() = 0;

private:
  // Note: gid_map only exists at the server
  map<int, relation_t> gid_map;
};

// This is a server_base object that executes all taskgraphs by first
// converting them into memgraphs.
struct server_mg_base_t : server_base_t {
  server_mg_base_t(
    communicator_t& c,
    allocator_settings_t s = allocator_settings_t::default_settings())
    : comm(c), alloc_settings(s),
      make_parallel_partialize_groups_(true),
      use_storage_(true)
  {}

  void set_parallel_partialize(bool new_val) {
    this->make_parallel_partialize_groups_ = new_val;
  }

  void set_use_storage(bool new_val) {
    this->use_storage_ = new_val;
  }

  void listen();

  void register_listen(string key, std::function<void(server_base_t*)> f);

  string get_registered_cmd() const {
    return write_with_ss(cmd_t::registered_cmd);
  }

  int get_max_tid();

  // get the max tid locally
  virtual int local_get_max_tid() const = 0;

  void shutdown() {
    broadcast_cmd(cmd_t::shutdown);
  }

  void execute(taskgraph_t const& taskgraph);

  // this must be called from all locations
  virtual void execute_memgraph(memgraph_t const& memgraph, bool for_remap) = 0;
  // the for_remap should be true if the computation is just a remap instead of
  // a general taskgraph

  dbuffer_t get_tensor(relation_t const& src_relation);

  void insert_relation(
    relation_t const& dst_relation,
    dbuffer_t src_tensor);
  void insert_relation_helper(
    remap_relations_t remap,
    map<int, buffer_t> data);

  void remap(remap_relations_t const& remap);

  virtual buffer_t local_copy_data(int tid) = 0;

  // return a location that exists at this compute-node
  virtual int local_candidate_location() const = 0;

  bool is_local_location(int loc) {
    return comm.get_this_rank() == loc_to_compute_node(loc);
  }
  virtual int loc_to_compute_node(int loc) const = 0;

  // Note: this remap is with respect to locations
  map<int, buffer_t> local_copy_source_data(remap_relations_t const& remap);

  void convert_remap_to_compute_node(remap_relations_t& remap);

  // server, client pairs {{{
  void execute_tg_server(taskgraph_t const& taskgraph);
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

  bool make_parallel_partialize_groups() {
    return make_parallel_partialize_groups_;
  }
public:
  communicator_t& comm;

  allocator_settings_t alloc_settings;

  bool make_parallel_partialize_groups_;
  bool use_storage_;

private:
  map<string, std::function<void(server_base_t*)>> listeners;

  enum class cmd_t {
    execute_tg = 0,
    remap,
    get_tensor,
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
};

std::ostream& operator<<(std::ostream& out, server_mg_base_t::cmd_t const& c);
std::istream& operator>>(std::istream& inn, server_mg_base_t::cmd_t& c);

