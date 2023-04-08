#include "graph.h"

int graph_t::insert_input(
  placement_t placement)
{
  return 0; // TODO
}

int graph_t::insert_input(
  partition_t partition)
{
  return this->insert_input(
    placement_t(placement_t(partition)));
}

int graph_t::insert_input(
  vector<uint64_t> shape)
{
  return this->insert_input(partition_t::singleton(shape));
}

int graph_t::insert_einsummable(
  placement_t placement,
  einsummable_t e,
  vector<int> inns)
{
  return 0; // TODO
}

int graph_t::insert_einsummable(
  partition_t partition,
  einsummable_t e,
  vector<int> inns)
{
  return this->insert_einsummable(placement_t(partition), e, inns);
}

int graph_t::insert_einsummable(
  einsummable_t e,
  vector<int> inns)
{
  vector<uint64_t> shape; // TODO: figure out the shape
  return this->insert_einsummable(partition_t::singleton(shape), e, inns);
}

int graph_t::insert_output(
  placement_t placement,
  int inn)
{
  return 0; // TODO
}

int graph_t::insert_output(
  partition_t partition,
  int inn)
{
  return this->insert_output(placement_t(partition), inn);
}

int graph_t::insert_output(
  vector<uint64_t> shape,
  int inn)
{
  return this->insert_output(partition_t::singleton(shape), inn);
}

void graph_t::set_outputs() {
  // TODO
}
