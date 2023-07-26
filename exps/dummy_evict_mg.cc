#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"

#include <fstream>

memgraph_t create_silly_evict_load(memgraph_t const& m0)
{
  memgraph_t m1(m0.num_compute_locs, m0.num_storage_locs, m0.storage_locs); 
  map<int, int> id_0_to_1;
  std::cout << "Node size: " << m0.nodes.size() << std::endl;
  for (int id = 0; id != m0.nodes.size(); ++id)
  {
    auto const& node = m0.nodes[id];
    std::cout << "M1 node size: " << m1.nodes.size() << std::endl;

    set<int> deps1;
    std::cout << "=========================BEGIN=======================" << std::endl;
    std::cout << "Node: " << id << std::endl;
    std::cout << "Dependencies: [" << std::endl;
    for (auto const& i : node.inns)
    {
      std::cout << "(" << i << ", "; 
      auto const& par = deps1.insert(id_0_to_1.at(i));
      std::cout << (*par.first) << "), ";
    }
    std::cout << "]" << std::endl;

    int id1; 
    if (node.op.is_inputmem())
    {
      memgraph_t::inputsto_t inputsto = {node.op.get_inputmem().loc, id, id};
      memgraph_t::op_t input_op(inputsto);
      int id0 = m1.insert(input_op, deps1);
      std::cout << "Inserted a node to m1. Insert returned id of: " << id0 << std::endl;

      memgraph_t::load_t load = {inputsto.as_stoloc(), node.op.get_inputmem().as_memloc()};
      memgraph_t::op_t load_op(load);
      deps1.insert(id0);
      id1 = m1.insert(load_op, deps1);
      std::cout << "Inserted a node to m1. Insert returned id of: " << id1 << std::endl;
    }
    else if(node.op.is_apply())
    {
      if (node.op.get_apply().is_einsummable())
      {
        int id0 = m1.insert(node.op, deps1);

        stoloc_t stoloc = {1, id};
        memloc_t mem = node.op.get_apply().mems[0].as_memloc(node.op.get_apply().loc);
        memgraph_t::evict_t evict = {mem, stoloc}; 
        set<int> deps2;
        deps2.insert(id0);
        int id2 = m1.insert(evict, deps2);

        memgraph_t::load_t load = {stoloc, mem};
        set<int> deps3;
        deps3.insert(id2);
        id1 = m1.insert(load, deps3);
      }
    }
    else 
    {
      id1 = m1.insert(node.op, deps1);
      std::cout << "Inserted a node to m1. Insert returned id of: " << id1 << std::endl;
    }

    std::cout << "Inserted node with id: " << id1 << std::endl;
    std::cout << "=========================END=========================" << std::endl;
    id_0_to_1.insert({id, id1});
  }

  std::cout << "M1 node size: " << m1.nodes.size() << std::endl;
  return m1;
}

void test_inputmem() {
  graph_constructor_t g;
  int inn = g.insert_input({20});
  int aaa = g.insert_formation(
    partition_t({ partdim_t::repeat(2, 10) }),
    inn);
  int bbb = g.insert_formation(
    partition_t({ partdim_t::repeat(1, 20) }),
    aaa);
  int ccc = g.insert_formation(
    partition_t({ partdim_t::repeat(2, 10) }),
    bbb);


  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  auto np = taskgraph.num_locs();

  auto [_2, _3, memgraph] = memgraph_t::make_without_evict(
    taskgraph, {},
    { allocator_strat_t::first, 1 }
  );

  memgraph_t m1 = create_silly_evict_load(memgraph);

  {
    std::cout << "silly1.gv" << std::endl;
    std::ofstream f("silly.gv");
    m1.print_graphviz(f);
  }
}

void test_apply() 
{
  int nlocs = 1;

  int ni = 5;
  int nj = 5;
  int nk = 5;

  int li = 2;
  int lj = 2;

  int rj = 2;
  int rk = 2;

  int ji = 2;
  int jj = 2;
  int jk = 2;

  int oi = 2;
  int ok = 2;

  graph_constructor_t g;
  dtype_t dtype = default_dtype();

  int lhs = g.insert_input(partition_t({
    partdim_t::split(ni, li),
    partdim_t::split(nj, lj) }));
  int rhs = g.insert_input(partition_t({
    partdim_t::split(nj, rj),
    partdim_t::split(nk, rk) }));

  int join = g.insert_einsummable(
    partition_t({
      partdim_t::split(ni, ji),
      partdim_t::split(nk, jk),
      partdim_t::split(nj, jj)
    }),
    einsummable_t::from_matmul(ni, nj, nk),
    {lhs, rhs});

  int out = g.insert_formation(
    partition_t({
      partdim_t::split(ni, oi),
      partdim_t::split(nk, ok)
    }),
    join);

  graph_t const& graph = g.graph;

  taskgraph_t taskgraph;

  auto pls = g.get_placements();
  for(int i = 0; i != pls.size(); ++i) {
    DOUT(i << " " << pls[i].partition);
  }

  // just set every block in every location to something
  // random
  set_seed(0);
  for(auto& placement: pls) {
    for(auto& loc: placement.locations.get()) {
      loc = runif(nlocs);
    }
  }

  auto [_0, _1, _taskgraph] = taskgraph_t::make(graph, pls);
  taskgraph = _taskgraph;

  auto [_2, _3, memgraph] = memgraph_t::make_without_evict(taskgraph);
  // ^ note that memsizes and allocat_settings not being provided

  memgraph_t m1 = create_silly_evict_load(memgraph);

  {
    std::ofstream f("silly2.gv");
    m1.print_graphviz(f);
    DOUT("printed mg.gv");
  }
}

int main(int argc, char** argv) {
  test_inputmem();
  test_apply();
}