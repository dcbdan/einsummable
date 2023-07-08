#include "../src/autoplace/gwise.h"

int main() {
  int nlocs = 3;

  uint64_t ni = 10001;
  uint64_t nj = 10002;
  uint64_t nk = 10003;

  graph_constructor_t g;
  int lhs = g.insert_input(partition_t({
    partdim_t::split(ni, 3),
    partdim_t::split(nj, 3)}));
  int rhs = g.insert_input(partition_t({
    partdim_t::split(nj, 3),
    partdim_t::split(nk, 3)}));
  int join = g.insert_einsummable(
    partition_t({
      partdim_t::split(ni, 3),
      partdim_t::split(nk, 3),
      partdim_t::split(nj, 3)}),
    einsummable_t::from_matmul(ni,nj,nk),
    {lhs, rhs});
  int out = g.insert_formation(
    partition_t({
      partdim_t::split(ni, 3),
      partdim_t::split(nk, 3)}),
    join);

  graph_t const& graph = g.graph;
  vector<placement_t> init_placements = g.get_placements();

  gwise_t gwise(nlocs, graph, init_placements);
  DLINEOUT(gwise.total_cost());
}
