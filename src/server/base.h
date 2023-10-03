#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"
#include "../einsummable/taskgraph.h"
#include "../einsummable/memgraph.h"
#include "../einsummable/relation.h"
#include "../einsummable/dbuffer.h"

#include "../engine/communicator.h"

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
  void execute(
    graph_t const& graph,
    vector<placement_t> const& placements);

  // Get a relation broadcast across the cluster and put it
  // onto node zero.
  virtual dbuffer_t get_tensor(relation_t const& src_relation) = 0;

  dbuffer_t get_tensor(int gid);

  relation_t const& get_relation(int gid) const;

  // Get a tensor here and partition it across the cluster
  // into managed data
  void insert_tensor(
    int gid,
    relation_t const& dst_relation,
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

  // it is error if any of the tids in these methods do not live on
  // _this_ rank.
  virtual void local_insert_tensors(map<int, buffer_t> data) = 0;
  virtual void local_erase_tensors(vector<int> const& tids) = 0;

  communicator_t& comm;

private:
  map<int, relation_t> gid_map;
};

// This is a server_base object but it executes all taskgraphs by first
// converting them into memgraphs.
struct server_mg_base_t : server_base_t {
  void listen();

  void register_listen(string key, std::function<void(server_base_t*)> f);

  string get_registered_cmd() const {
    return write_with_ss(cmd_t::registered_cmd);
  }

  int get_max_tid();

  void shutdown() {
    broadcast_cmd(cmd_t::shutdown);
  }

  void execute(taskgraph_t const& taskgraph);

  // this must be called from all locations
  virtual void execute_memgraph(memgraph_t const& memgraph) = 0;

  dbuffer_t get_tensor(relation_t const& src_relation);

  void insert_relation(
    relation_t const& dst_relation,
    dbuffer_t src_tensor);

  void remap(remap_relations_t const& remap);

  virtual buffer_t local_copy_data_at(memsto_t const& loc) = 0;

  buffer_t local_copy_data(int tid);

  map<int, buffer_t> local_copy_source_data(remap_relations_t const& remap);

  // mapping from tid to where the data lives
  map<int, memsto_t> data_locs;

  allocator_settings_t alloc_settings;

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

  virtual vector<uint64_t> recv_mem_sizes() = 0;
  virtual void send_mem_sizes() = 0;

  virtual void rewrite_data_locs_server(map<int, memstoloc_t> const& out_tg_to_loc) = 0;
  virtual void rewrite_data_locs_client() = 0;
  // }}}

  static
  vector<vector<std::array<int, 2>>>
  create_storage_remaps(
    int world_size,
    map<int, memstoloc_t> const& full_data_locs,
    map<int, memstoloc_t> const& inn_tg_to_loc);

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

