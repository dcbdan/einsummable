// #include "../src/autoplace/relationwise.h"
#include "../src/base/copyregion.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/autoplace/apart.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/autoplace/alocate.h"
#include "../src/base/placement.h"

#include <fstream>


using tensor_t     = graph_writer_t::tensor_t;

graph_t make_graph_ff_simple(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});

  tensor_t w1 = writer.input({hidden, dim});
  tensor_t w2 = writer.input({dim, hidden});
  tensor_t w3 = writer.input({hidden, dim});

  tensor_t w1t = w1.transpose(0, 1);
  tensor_t w2t = w2.transpose(0, 1);
  tensor_t w3t = w3.transpose(0, 1);

  scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

  tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

  tensor_t b = writer.matmul(x, w3t) ;

  tensor_t c = writer.mul(a, b);

  writer.matmul(c, w2t).save_inplace();

  return writer.get_graph();
}



vector<placement_t> create_same_loc_placements(vector<partition_t> const& parts, int num_locs)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  for(auto const& part: parts) {
    // std::cout <<  "part: " << part  <<  std::endl;
    // for(int i = 0; i <  part.num_parts(); i++){
    //   std::cout << part.total_shape()  <<  std::endl;
    // }
    
    ret.emplace_back(part);
  }
  return ret;
}



vector<placement_t> pre_assign_loc(vector<partition_t> const& parts, int num_locs, vector<int> placement, graph_t const& graph)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  int i = 0;
  for(auto const& part: parts) {

    std::cout << "part: " << part  <<  std::endl;
    ret.emplace_back(part);
    for(int& loc: ret.back().locations.get()) {
      loc = placement[i];
      i+=1;
    }
  }
  std::cout << "Total number of nodes: " << i << std::endl;
  return ret;
}

int main(){

    // uint64_t ni = 10001;
    // uint64_t nj = 10002;
    // uint64_t nk = 10003;
    
    // vector<partition_t> parts;
    // graph_constructor_t g;
    // int lhs = g.insert_input(partition_t({
    // partdim_t::split(ni, 1),
    // partdim_t::split(nj, 4)}));

    // parts.push_back(partition_t({
    // partdim_t::split(ni, 1),
    // partdim_t::split(nj, 4)}));

    // int rhs = g.insert_input(partition_t({
    // partdim_t::split(nj, 2),
    // partdim_t::split(nk, 3)}));

    // parts.push_back(partition_t({
    // partdim_t::split(nj, 2),
    // partdim_t::split(nk, 3)}));

    // int join = g.insert_einsummable(
    // partition_t({
    //     partdim_t::split(ni, 1),
    //     partdim_t::split(nk, 2),
    //     partdim_t::split(nj, 3)}),
    // einsummable_t::from_matmul(ni,nj,nk),
    // {lhs, rhs});

    // parts.push_back(partition_t({
    //     partdim_t::split(ni, 1),
    //     partdim_t::split(nk, 2),
    //     partdim_t::split(nj, 3)}));

    int nlocs = 16;
    graph_t const& graph = make_graph_ff_simple(1000,1000,1000);

    std::ofstream f("my_graph.gv");
    graph.print_graphviz(f);

    vector<partition_t> parts = apart01(graph,nlocs);
    // vector<int> rl_place = {1, 7, 2, 6, 3, 2, 4, 1, 5, 6, 7, 2, 5, 3, 7, 3, 2, 0, 7, 1, 7, 2, 6, 3,
    //     2, 4, 1, 5, 6, 7, 2, 3, 3, 7, 3, 1, 6, 2, 1, 3, 0, 4, 1, 5, 6, 7, 0, 3,
    //     3, 7, 3, 1, 6, 3, 6, 3, 0, 4, 1, 5, 6, 7, 2, 5, 3, 7, 3, 1, 6, 3, 1, 3,
    //     0, 3, 1, 5, 6, 7, 0, 5, 3, 7, 3, 1, 6, 3, 5, 3, 0, 3, 0, 5, 6, 7, 0, 5,
    //     5, 7, 3};
    std::vector<placement_t> placements = create_same_loc_placements(parts,nlocs);
    // // vector<placement_t> placements = alocate01(graph, parts,8,100);
    // // for(auto const& p: placements){
    // //   const vtensor_t i = p.locations;
    // //   std::cout << i << std::endl;
    // // }
    // vector<placement_t> placements = pre_assign_loc(parts, 8, rl_place, graph);
    
    
    
    
    auto [inn_gid_to_tids, out_gid_to_tids, part_graph] = taskgraph_t::make(graph, placements);
    // std::cout << "part_graph.nodes.size(): " << part_graph.nodes.size() << std::endl;
    // int xxx = graph.nodes.size();
    // std::cout << xxx << std::endl;
    // //std::cout << "graph.nodes.size(): " << (graph.nodes.size()) << std::end;
    // // vector<placement_t> einsum_placements = part_graph.get_placements();

    std::ofstream f1("my_task_graph.gv");
    part_graph.print_graphviz(f1);


    // server_base_t::execute_graph()

    

    relationwise_t gwise(graph, parts);

    std::map<rid_t, std::set<rid_t>> xin_rl_graph_dict;
    std::map<rid_t, int> key_node_idx_dict;
    std::vector<int> input_list;
    std::vector<int> output_list;

    // std::cout << "number of nodes: " << graph.nodes.size() << std::endl;
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
        // std::cout << "gid: " << gid << std::endl;
        auto const& ginfo = gwise.ginfos[gid];
        int nbid = ginfo.locations.size();
        for(int bid = 0; bid != nbid; ++bid) {
            auto const& join = ginfo.joins[bid];
            for(auto const& rid: join.deps){
              // std::cout << "   J " << bid << ": " << rid << std::endl;
              rid_t rid_key {gid,bid};
              xin_rl_graph_dict[rid].insert(rid_key);
              key_node_idx_dict[rid] = 0;
              key_node_idx_dict[rid_key] = 0;
            }
        }
        if(ginfo.refis) {
            auto const& refis = ginfo.refis.value();
            for(int bid = 0; bid != refis.size(); ++bid) {
                auto const& refi = refis[bid];
                for(auto const& unit: refi.units) {
                    // std::cout << "  R " << bid << ": " << unit.deps << std::endl;

                    // rid_t rid_key {gid,bid};
                    // if (xin_rl_graph_dict.find(rid_key) == xin_rl_graph_dict.end()){
                    //     xin_rl_graph_dict[rid_key] = {};
                    // }
                    // xin_rl_graph_dict[rid_key].insert(unit.deps.begin(), unit.deps.end());
                }
            }
        }
    }

    // gwise.print_info();
    
    int cnt = 0;
    for(auto const&[key, _]: key_node_idx_dict){
        key_node_idx_dict[key] = cnt;
        cnt += 1;
    }

    for(auto const&[key, value]: xin_rl_graph_dict){
        int node_idx = key_node_idx_dict[key];
        for(auto const&val: value){
          int output_idx = key_node_idx_dict[val];
          input_list.push_back(node_idx);
          output_list.push_back(output_idx);
          std::cout << key << ":" << node_idx << " --> " << val << ":" << output_idx << std::endl;
        }
    }
  
  std::ofstream graph_feature_file;
  graph_feature_file.open("rl_graph_from_two_layer_graph.txt");
  graph_feature_file << cnt << std::endl;
  for(auto const& inn: input_list){
    graph_feature_file << inn;
    graph_feature_file << ", ";
  }
  graph_feature_file << std::endl;

  for(auto const& out: output_list){
    graph_feature_file << out;
    graph_feature_file << ", ";
  }
  graph_feature_file << std::endl;

  for(auto const& part: parts) {
    std::cout <<  "part: " << part  <<  std::endl;
    for(int i = 0; i <  part.num_parts(); i++){
      vector<long unsigned int> part_shape = part.total_shape();
      if(part_shape.size() < 3){
        part_shape.push_back(1);
      }
      std::cout << part_shape <<  std::endl;
      graph_feature_file << part_shape << std::endl;
    }
  }

}
