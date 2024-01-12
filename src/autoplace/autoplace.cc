#include "autoplace.h"

#include "apart.h"
#include "alocate.h"

vector<placement_t> autoplace01(
  graph_t const& graph,
  autoplace_config_t const& config)
{
  vector<partition_t> parts = apart01(
    graph,
    config.n_compute,
    config.max_branching,
    config.discount_input_factor,
    config.search_space);

  return alocate01(
    graph,
    parts,
    config.n_compute,
    config.flops_per_byte_moved);
}
