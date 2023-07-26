#pragma once
#include "../../base/setup.h"

#include "mpi_class.h"
#include "executetg.h"

#include "../../einsummable/relation.h"

// A tg manager updates a map<int, buffer_t> state
// from a client node at rank zero. All the other nodes
// cooperate to do the various updates.
struct tg_manager_t {
  tg_manager_t(mpi_t* mpi, execute_taskgraph_settings_t const& settings);

  // this should be called by all non-zero rank locations
  void listen();

  void register_listen(string key, std::function<void(tg_manager_t&)> f);

  // Should only be called by rank zero and when all other
  // ranks are listening {{{
  void execute(taskgraph_t const& taskgraph);

  // Get a relation broadcast across the cluster and put it
  // onto node zero. Don't modify this->data
  dbuffer_t unpartition(relation_t const& src_relation);

  // Get a tensor here and partition it across the cluster
  // into this->data
  void partition_into_data(
    relation_t const& dst_relation,
    dbuffer_t src_tensor);

  // repartitions the relations stored in data,
  // modifying this->data in the process
  // Note: anything not in the dst remap may get
  //       deleted!
  void remap_data(remap_relations_t const& remap);

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
    unpartition,
    partition_into_data,
    remap_data,
    max_tid,
    registered_cmd,
    shutdown
  };

  static vector<string> const& cmd_strs() {
    static vector<string> ret {
      "execute", "unpartition", "partition_into_data",
      "remap_data", "max_tid", "registered_cmd", "shutdown"
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

  void copy_into_data(
    map<int, buffer_t>& tmp,
    remap_relations_t const& remap);

  map<string, std::function<void(tg_manager_t&)>> listeners;
};

std::ostream& operator<<(std::ostream& out, tg_manager_t::cmd_t const& c);
std::istream& operator>>(std::istream& inn, tg_manager_t::cmd_t& c);

