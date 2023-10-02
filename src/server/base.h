#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"
#include "../einsummable/taskgraph.h"
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
  virtual void partition_into(
    relation_t const& dst_relation,
    dbuffer_t src_tensor) = 0;

  // repartitions the relations stored in managed data
  // Note: anything not in the dst remap may get
  //       deleted!
  virtual void remap(remap_relations_t const& remap) = 0;

  // Note: gid_map will only contain the dst tids after this
  void remap_gids(vector<tuple<int,int>> const& remap);

  // Get the max tid across all data objects on all ranks.
  // Useful for creating new relations that won't overwrite
  // existing data
  virtual int get_max_tid() = 0;

  virtual void shutdown() = 0;
  // }}}

  virtual communicator_t& get_communicator() = 0;

  // TODO: what is this for?
  virtual string get_registered_cmd() const = 0;

  // it is error if any of the tids in these methods do not live on
  // _this_ rank.
  virtual void insert_tensors(map<int, buffer_t> data) = 0;
  virtual void erase_tensors(vector<int> const& tids) = 0;

private:
  map<int, relation_t> gid_map;
};


