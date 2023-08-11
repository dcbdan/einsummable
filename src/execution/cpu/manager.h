#pragma once
#include "../../base/setup.h"

#include "mpi_class.h"
#include "executetg.h"
#include "executemg.h"

#include "../../einsummable/relation.h"
#include "einsummable.pb.h"

struct manager_base_t {
  virtual void listen() = 0;

  // Note: this should only be called on ranks not equal to zero.
  virtual void register_listen(string key, std::function<void(manager_base_t*)> f) = 0;

  // Should only be called by rank zero and when all other
  // ranks are listening {{{
  virtual void execute(taskgraph_t const& taskgraph) = 0;

  // Get a relation broadcast across the cluster and put it
  // onto node zero.
  virtual dbuffer_t get_tensor(relation_t const& src_relation) = 0;

  // Get a tensor here and partition it across the cluster
  // into managed data
  virtual void partition_into(
    relation_t const& dst_relation,
    dbuffer_t src_tensor) = 0;

  // repartitions the relations stored in managed data
  // Note: anything not in the dst remap may get
  //       deleted!
  virtual void remap(remap_relations_t const& remap) = 0;

  // Update the kernel manager across all nodes
  virtual void update_kernel_manager(taskgraph_t const& tg) = 0;

  virtual void custom_command(string key) = 0;

  // Get the max tid across all data objects on all ranks.
  // Useful for creating new relations that won't overwrite
  // existing data
  virtual int get_max_tid() = 0;

  virtual void shutdown() = 0;
  // }}}
};

// TODO: There is a lot code duplication in the implementation of tg_manager_t
//       and mg_manager_t. When it becomes clear how these managers will be used,
//       consolidate into manager_base_t.

// A tg manager updates a map<int, buffer_t> state
// from a client node at rank zero. All the other nodes
// cooperate to do the various updates.
struct tg_manager_t : public manager_base_t {
  tg_manager_t(mpi_t* mpi, execute_taskgraph_settings_t const& settings);

  void listen();

  void register_listen(string key, std::function<void(manager_base_t*)> f);

  // Should only be called by rank zero and when all other
  // ranks are listening {{{
  void execute(taskgraph_t const& taskgraph);

  // Get a relation broadcast across the cluster and put it
  // onto node zero. Don't modify this->data
  dbuffer_t get_tensor(relation_t const& src_relation);

  // Get a tensor here and partition it across the cluster
  // into this->data
  void partition_into(
    relation_t const& dst_relation,
    dbuffer_t src_tensor);

  // repartitions the relations stored in data,
  // modifying this->data in the process
  // Note: anything not in the dst remap may get
  //       deleted!
  void remap(remap_relations_t const& remap);

  void update_kernel_manager(taskgraph_t const& tg);

  void custom_command(string key);

  // Get the max tid across all data objects on all ranks.
  // Useful for creating new relations that won't overwrite
  // existing data
  int get_max_tid();

  void shutdown();
  // }}}

  static string get_registered_cmd() {
    return write_with_ss(cmd_t::registered_cmd);
  }

  map<int, buffer_t> data;

  mpi_t* mpi;
  execute_taskgraph_settings_t settings;
  kernel_manager_t kernel_manager;

private:
  // cmd = command, str = string

  enum class cmd_t {
    execute = 0,
    update_km,
    unpartition,
    partition_into,
    remap,
    max_tid,
    registered_cmd,
    shutdown
  };

  static vector<string> const& cmd_strs() {
    static vector<string> ret {
      "execute", "update_km", "unpartition", "partition_into",
      "remap", "max_tid", "registered_cmd", "shutdown"
    };
    return ret;
  }

  friend std::ostream& operator<<(std::ostream& out, cmd_t const& c);
  friend std::istream& operator>>(std::istream& inn, cmd_t& c);

  void broadcast_cmd(cmd_t const& cmd);
  void broadcast_str(string const& str);

  // only recvs from rank 0
  cmd_t recv_cmd();

  void _execute(taskgraph_t const& tg);

  void _broadcast_es(vector<std::unordered_set<einsummable_t>> const& es);
  void _update_km(std::unordered_set<einsummable_t> const& es);
  void _update_km(es_proto::EinsummableList const& es);

  void copy_into_data(
    map<int, buffer_t>& tmp,
    remap_relations_t const& remap);

  map<string, std::function<void(manager_base_t*)>> listeners;
};

std::ostream& operator<<(std::ostream& out, tg_manager_t::cmd_t const& c);
std::istream& operator>>(std::istream& inn, tg_manager_t::cmd_t& c);

struct mg_manager_t : public manager_base_t {
  mg_manager_t(
    mpi_t* mpi,
    execute_memgraph_settings_t const& exec_sts,
    uint64_t memory_size,
    allocator_settings_t alloc_sts = allocator_settings_t::default_settings());

  // this should be called by all non-zero rank locations
  void listen();

  void register_listen(string key, std::function<void(manager_base_t*)> f);

  // Should only be called by rank zero and when all other
  // ranks are listening {{{

  // Compile a taskgraph into a memgraph and execute the
  // memgraph
  void execute(taskgraph_t const& taskgraph);
  // (returning the memgraph for debugging)

  void execute(memgraph_t  const& memgraph);

  // Get a relation broadcast across the cluster and put it
  // onto node zero. Don't modify data owned by this
  dbuffer_t get_tensor(relation_t const& src_relation);

  // Get a tensor here and partition it across the cluster
  void partition_into(
    relation_t const& dst_relation,
    dbuffer_t src_tensor);

  void remap(remap_relations_t const& remap);

  void update_kernel_manager(taskgraph_t const& tg);
  void update_kernel_manager(memgraph_t const& mg);

  void custom_command(string key);

  // Get the max tid across all data objects on all ranks.
  // Useful for creating new relations that won't overwrite
  // existing data
  int get_max_tid();

  void shutdown();
  // }}}

  map<int, memsto_t> data_locs;

  buffer_t mem;
  storage_t storage;

  mpi_t* mpi;
  execute_memgraph_settings_t exec_settings;
  allocator_settings_t alloc_settings;
  kernel_manager_t kernel_manager;

private:
  enum class cmd_t {
    execute_tg = 0,
    execute_mg,
    update_km,
    unpartition,
    partition_into,
    remap,
    max_tid,
    registered_cmd,
    shutdown
  };

  static vector<string> const& cmd_strs() {
    static vector<string> ret {
      "execute_tg", "execute_mg", "update_km",
      "unpartition", "partition_into",
      "remap", "max_tid", "registered_cmd", "shutdown"
    };
    return ret;
  }

  friend std::ostream& operator<<(std::ostream& out, cmd_t const& c);
  friend std::istream& operator>>(std::istream& inn, cmd_t& c);

  void broadcast_cmd(cmd_t const& cmd);
  void broadcast_str(string const& str);

  // only recvs from rank 0
  cmd_t recv_cmd();

  void _execute_mg(memgraph_t const& mg);

  void _broadcast_es(vector<std::unordered_set<einsummable_t>> const& es);
  void _update_km(std::unordered_set<einsummable_t> const& es);
  void _update_km(es_proto::EinsummableList const& es);

  buffer_t get_copy_of_data(int tid);

  map<int, buffer_t> _unpartition(remap_relations_t const& remap);

  void _remap_into_here(remap_relations_t const& remap, map<int, buffer_t> data);

  map<string, std::function<void(manager_base_t*)>> listeners;
};

std::ostream& operator<<(std::ostream& out, mg_manager_t::cmd_t const& c);
std::istream& operator>>(std::istream& inn, mg_manager_t::cmd_t& c);

