#include "../llama/misc.h"
#include "../llama/modules.h"
#include "../llama/builder.h"
#include "../llama/reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

communicator_t gpu_mg_server_t::null_comm;

template <typename T>
void print_graphviz(T const& obj, string filename)
{
  std::ofstream f(filename);
  obj.print_graphviz(f);
  DOUT("printed " << filename);
}

int main() {
  uint64_t ni = 100;

  graph_writer_t writer;

  using tensor_t = graph_writer_t::tensor_t;

  tensor_t lhs1 = writer.input({ni,ni});
  tensor_t rhs1 = writer.input({ni,ni});

  tensor_t lhs2 = writer.input({ni,ni});
  tensor_t rhs2 = writer.input({ni,ni});

  tensor_t out1 = writer.matmul(lhs1, rhs1);
  tensor_t out2 = writer.matmul(lhs2, rhs2);

  tensor_t out = writer.matmul(out1, out2).save();

  auto const& graph = writer.get_graph();

  print_graphviz(graph, "g.gv");

  int num_gpus = 1;
  int num_computes_per_loc = 1;

  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);


  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);

  print_graphviz(taskgraph, "tg.gv");

  vector<uint64_t> mem_sizes(num_gpus, 500000);

  vector<allocator_t> allocs;
  for (auto m : mem_sizes) {
    allocs.emplace_back(m);
  }
  map<int, memstoloc_t> empty_map;
  memgraph_make_state_t state(taskgraph, {}, allocs, empty_map, 1, 0, false);
  
  for(int id = 0; id != taskgraph.nodes.size(); ++id)
  {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input())
    {
      if(node.outs.size() == 0 && !node.is_save)
      {
          throw std::runtime_error(
            "This is goofy: an input to memgraph is not used or saved."
            " Call this again after pruning inputs that don't get used"
            " or saved.");
      }

        // It could be the case that the used initialized the input
      if(!state.input_has_been_initialized(id))
      {
        state.initialize_input(id);
      }
    }
  }
  state.process(order_taskgraph(taskgraph));

  // auto [_2, _3, memgraph] = memgraph_t::make(taskgraph, {},  mem_sizes);

  std::cout << "Memgraph nodes before move: " << std::to_string(state.memgraph.nodes.size()) << std::endl;

  int compute_task_id = 6;
  if (state.memgraph.nodes[state.task_node_to_mem_node[_which_node_t{.task_id=compute_task_id}]].op.is_apply()){
    uint64_t output_mem_offset = state.memgraph.nodes[state.task_node_to_mem_node[_which_node_t{.task_id=compute_task_id}]].op.get_output_mem().offset;
    vector<tuple<uint64_t /*offset*/, uint64_t /*size*/, int/*loc*/, uint64_t /*new_offset*/, int /*new_loc*/, int>> loc_map;
    loc_map.emplace_back(output_mem_offset, 40000, 0, 40000, 0, compute_task_id);
    loc_map.emplace_back(output_mem_offset, 40000, 0, 0, 0, compute_task_id);
    loc_map.emplace_back(output_mem_offset, 40000, 0, 80000, 0, compute_task_id);
    loc_map.emplace_back(output_mem_offset, 40000, 0, 120000, 0, compute_task_id);
    state.move_tensors(loc_map);

    // Create server and exec memgraph
    gpu_mg_server_t serv(mem_sizes, 10e6);
    void *gpu_mem = serv.mems[0];

    uint64_t tensor_size = ni * ni * 4;
    buffer_t input_buf = make_buffer(tensor_size);
    dbuffer_t input_dbuf(dtype_t::f32, input_buf);
    input_dbuf.ones();

    for (int i = 0; i < 4; i++) {
        cudaError_t error = cudaMemcpy(gpu_mem + (i * tensor_size), input_dbuf.raw(), tensor_size, cudaMemcpyHostToDevice);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed");
        }
    }

    DOUT(gpu_mem + output_mem_offset);

    serv.execute_memgraph(state.memgraph, false, {});
    cudaError_t error = cudaMemcpy(input_dbuf.raw(), gpu_mem + output_mem_offset, tensor_size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed");
    }
    DOUT(input_dbuf.get(0).str());


    serv.execute_memgraph(state.memgraph, false, {});
    error = cudaMemcpy(input_dbuf.raw(), gpu_mem + output_mem_offset, tensor_size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed");
    }
    DOUT(input_dbuf.get(0).str());

    //std::cout << input_dbuf << std::endl;
  }



  std::cout << "Memgraph nodes after move: " << std::to_string(state.memgraph.nodes.size()) << std::endl;

  print_graphviz(state.memgraph, "mgmake.gv");

  
}