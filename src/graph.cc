#include "graph.h"

int graph_t::insert_input(
  placement_t placement)
{
  return this->insert(
    input_t{ .shape = placement.total_shape()},
    {},
    placement);
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
  if(e.inns.size() != inns.size()) {
    throw std::runtime_error("did not get expected number of inputs");
  }

  auto expected_inn_shapes = e.inn_shapes();
  for(int i = 0; i != inns.size(); ++i) {
    if(!vector_equal(expected_inn_shapes[i], out_shape(inns[i]))) {
      throw std::runtime_error("shapes do not match: insert einsummable");
    }
  }

  return this->insert(
    e,
    inns,
    placement);
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
  return this->insert_einsummable(
    partition_t::singleton(e.join_shape),
    e,
    inns);
}

int graph_t::insert_output(
  placement_t placement,
  int inn)
{
  if(!vector_equal(placement.total_shape(), out_shape(inn))) {
    throw std::runtime_error("invalid shape: insert_output");
  }

  return this->insert(
    output_t { .shape = placement.total_shape() },
    {inn},
    placement);
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
  int num_nodes_time_zero = nodes.size();
  for(int i = 0; i != num_nodes_time_zero; ++i) {
    // All non-output nodes must have an outgoing edge.
    // If any non-output nodes don't have that, set the output
    // of the node.
    node_t const& n = nodes[i];
    if(!n.op.is_output()) {
      if(n.outs.size() == 0) {
        // Create a placement that matches closely to the
        // output tensor of the input node
        this->insert_output(
          placement_t::join_to_out(n.placement, n.op.out_rank()),
          i);
      }
    }
  }
}

vector<uint64_t> graph_t::out_shape(int id) {
  return nodes[id].op.out_shape();
}

vector<int> graph_t::get_order() const {
  // Because of the way the graph is constructed,
  // it must be the case that a valid ordering of the compute
  // graph is 0,1,2,...
  vector<int> ret(nodes.size());
  std::iota(ret.begin(), ret.end(), 0);
  return ret;
}

int graph_t::insert(
  op_t const& op,
  vector<int> inns,
  placement_t placement)
{
  int ret = nodes.size();
  nodes.push_back(node_t {
    .op = op,
    .inns = inns,
    .outs = {},
    .placement = placement
  });

  for(auto inn: inns) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

