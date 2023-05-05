#include "memgraph.h"

memgraph_t::memgraph_t(
  int nl, int nc,
  vector<int> cs,
  vector<uint64_t> const& ms)
  : num_compute_locs(nl), num_cache_locs(nc),
    cache_locs(cs), mem_sizes(ms)
{}

tuple<
  map<int, mem_t>,
  memgraph_t >
memgraph_t::make(
  taskgraph_t const& taskgraph,
  vector<uint64_t> const& mem_sizes,
  vector<int> const& which_cache)
{
  // TODO: verify each op can fit in memory
  // TODO: verify all save nodes can fit in memory

  int const n_compute_locs = taskgraph.num_locs();
  if(mem_sizes.size() != n_compute_locs) {
    throw std::runtime_error("incorrect mem length: memgraph_t::make")
  }
  if(which_cache.size() != n_compute_locs) {
    throw std::runtime_error("incorrect which cache length: memgraph_t::make")
  }

  int nlocs = n_compute_locs;
  for(int const& cache_loc: which_cache) {
    if(cache_loc < nlocs) {
      throw std::runtime_error("invalid cache location");
    }
    nlocs = std::max(nlocs, cache_loc + 1);
  }

  int n_cache_locs = nlocs - n_compute_locs;

  memgraph_t memgraph(n_compute_locs, n_cache_locs, mem_sizes);

  // TODO: what ordering to choose? An ordering should be chosen
  //       such that there is enough parallel work but not full
  //       breadthfirst order which would use too much memory.
  //       Chances are a "parallel-ness" parameter will need to
  //       be chosen.
  //
  //       For now: just get any order at all.
  vector<int> taskgraph_ordering = taskgraph.get_order();
  // TODO: partialize nodes are weird: they consist of several
  //       touches. So maybe what you want is get also an ordering
  //       of the partialize ops.. But it's probably fine to assume
  //       of partialize ops happen as a single operation in this context.

  for(int const& tid: taskgraph_ordering) {
    auto const& task_node = taskgraph.nodes[tid];
    // TODO
    // Can we allocate this output?
    //   If so, allocate it.
    //   Otherwise, do an eviction of a tensor
    //   used along time ago and not used soon.
    // Add this task node to the memgraph
    // Now that the task node has been added,
    //   can we add some deletes to memgraph?
    //   Then do so.
    // If some memory has been freed up, see if
    //   we can load any evicted tensors, in the
    //   order that they'll be used next
  }

  // TODO: Verify that all loaded tensors are used atleast once TODO TODO TODO

  // TODO: Get taskid to memid mapping for all save nodes.
  //       Also, mark save nodes as save if that wasn't done before here
  map<int, int> save_taskid_to_memid;

  return {save_taskid_to_memid, memgraph};
}

