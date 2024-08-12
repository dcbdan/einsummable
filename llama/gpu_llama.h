#pragma once

#include "misc.h"
#include "modules.h"

#include "../src/server/gpu/server.h"

#include "../src/misc/checkpoint.h"
#include "../src/misc/update.h"

#include "../src/engine/repartition.h"
#include "../src/autoplace/autoplace.h"

#include <cuda_runtime_api.h>
#include <stdexcept>


enum class llama_size_t {B7 = 1, B13 = 2, B30 = 4, B65 = 8};

struct graph_setup_t {
  model_args_t margs;
  graph_t full_graph;
  checkpoint_graphs_t checkpoint_graphs;
  updater_desc_t updater_desc;
  vector<tuple<int, int>> old_news;
  vector<tuple<int, fill_t>> init_fills;
  int embeddings_id;
  int predictions_id;
  int labels_id;
  int loss_id;
  int full_freqs_cis_id;
  vector<tuple<string, int>> model_weight_map;

  vector<uint64_t> get_shape(int id) const {
    return full_graph.out_shape(id);
  }
};


// struct tensor_handler_t {
//   graph_t full_graph;
//   checkpoint_graphs_t checkpoint_graphs;
//   map<int, int> gid_to_init_cg;
//   checkpoint_taskgraphs_t checkpoint_taskgraphs;
//   vector<memgraph_make_state_t> checkpoint_memgraphs;
//   map<string, int> model_weight_map;
//   gpu_mg_server_t* server;

//   tensor_handler_t(graph_t full_graph,
//   checkpoint_graphs_t checkpoint_graphs,
//   map<int, int> gid_to_init_cg,
//   checkpoint_taskgraphs_t checkpoint_taskgraphs,
//   vector<memgraph_make_state_t> checkpoint_memgraphs,
//   map<string, int> model_weight_map,
//   gpu_mg_server_t* server) : checkpoint_graphs(checkpoint_graphs), full_graph(full_graph), gid_to_init_cg(gid_to_init_cg),
//   checkpoint_taskgraphs(checkpoint_taskgraphs), checkpoint_memgraphs(checkpoint_memgraphs), model_weight_map(model_weight_map),
//   server(server) {}

//   dbuffer_t get_tensor(string weight) {
//     if (model_weight_map.find(weight) == model_weight_map.end()) {
//       throw std::runtime_error("Requested weight does not exist");
//     }
//     return get_tensor(model_weight_map[weight]);
//   }

//   dbuffer_t get_tensor(int gid) {
//     if (gid_to_init_cg.find(gid) == gid_to_init_cg.end()) {
//       throw std::runtime_error("Unavailable gid entered to get_tensor");
//     }
//     int cg_gid = gid_to_init_cg[gid];
//     relation_t& rel = checkpoint_taskgraphs.infos[0].init_rel.at(cg_gid);

//     vector<uint64_t> shape = full_graph.out_shape(gid);
//     dtype_t dtype = full_graph.out_dtype(gid);
//     // relation_t rel = relation_t::make_singleton(dtype, shape, 0);
//     // relation_t new_rel = relation_t::make_singleton(dtype, shape, 0);
//     relation_t new_rel = rel.as_singleton(0);

//     remap_relations_t remap;
//     remap.insert(rel, new_rel);

//     map<int, buffer_t> data;
//     for (auto &tid : rel.tids.get()) {
//       int mid = checkpoint_memgraphs[0].task_tensor_to_mem_node[tid];

//       auto &node = checkpoint_memgraphs[0].memgraph.nodes[mid];

//       buffer_t out;

//       if (!node.op.is_inputmem() && !node.op.is_inputsto()) {
//         throw std::runtime_error("Tensor to get does not correspond to inputmem or inputsto node");
//       }
//       if (node.op.is_inputmem()) {
//         auto& input = node.op.get_inputmem();
//         out = make_buffer(input.size);

//         cudaError_t error = cudaMemcpy(out->raw(), increment_void_ptr(server->mems[input.loc], input.offset), input.size, cudaMemcpyDeviceToHost);
//         if(error != cudaSuccess) {
//           DOUT(cudaGetErrorString(error));
//           throw std::runtime_error("cudaMemcpy failed");
//         }
//       } else {
//         auto& input = node.op.get_inputsto();
//         out = server->get_storage_buf(input.storage_id);
//       }

//       data[tid] = out;
//     }

//     repartition(server->null_comm, remap, data, nullptr);
//     return dbuffer_t(dtype, data[0]);
//   }

//   void load_tensor(int gid, dbuffer_t d) {
//     if (gid_to_init_cg.find(gid) == gid_to_init_cg.end()) {
//       throw std::runtime_error("Unavailable gid entered to get_tensor");
//     }
//     int cg_gid = gid_to_init_cg[gid];
//     relation_t& new_rel = checkpoint_taskgraphs.infos[0].init_rel.at(cg_gid);
    
//     relation_t rel = new_rel.as_singleton(0);
//     remap_relations_t remap;
//     remap.insert(rel, new_rel);

//     map<int, buffer_t> data;
//     data[0] = d.data;
//     repartition(server->null_comm, remap, data, nullptr);

//     for (auto &[tid, buf] : data) {
      
//       if (checkpoint_memgraphs[0].input_tid_to_data.find(tid) == checkpoint_memgraphs[0].input_tid_to_data.end()) {
//         throw std::runtime_error("Tid " + std::to_string(tid) + " does not exist in input_tid_to_data");
//       } 

//       memstoloc_t inn = checkpoint_memgraphs[0].input_tid_to_data[tid];

//       if (inn.is_memloc()) {
//         auto& input = inn.get_memloc();

//         cudaError_t error = cudaMemcpy(increment_void_ptr(server->mems[input.loc], input.offset), buf->raw(), input.size, cudaMemcpyDefault);
//         if(error != cudaSuccess) {
//           DOUT(cudaGetErrorString(error));
//           throw std::runtime_error("cudaMemcpy failed");
//         }
//       } else {
//         auto& input = inn.get_stoloc();
//         server->set_storage_buf(buf, input.id);
//       }
//     }
//   }
// };


struct gpu_llama_t {
    gpu_llama_t(uint64_t mem_size, uint64_t storage_size, llama_size_t llama_size = llama_size_t::B7);

    vector<uint64_t> buffer_sizes = {125lu * 100lu * 1000lu * 1000lu};
    uint64_t storage_size;
    bool is_adamw = true;

    // Llama args
    int seed = -1;
    int batch_size = 1;
    int lora_rank = 32;
    int max_n_layers = -1;
    uint64_t seq_len = 4096;
    float learning_rate = 1e-9f;
    dtype_t dtype = dtype_t::f32;
    updater_desc_t updater_desc;
    llama_size_t llama_size = llama_size_t::B7;
    model_args_t margs = model_args_t::llama((int) llama_size, batch_size);

    // Metadata
    int embeddings_id = -1;
    int predictions_id = -1;
    int labels_id = -1;
    int loss_id = -1;
    int full_freqs_cis_id = -1;

    // Graphs
    graph_setup_t graph;
    std::unique_ptr<checkpoint_taskgraphs_t> taskgraphs;
    vector<std::unique_ptr<memgraph_make_state_t>> states;
    vector<map<int, memstoloc_t>> input_tid_to_datas;
    // std::unique_ptr<tensor_handler_t> tensor_handler;

    communicator_t communicator;
    std::unique_ptr<gpu_mg_server_t> server;

    vector<vector<std::array<int, 2>>> remaps;
    map<int, int> gid_to_init_cg;
    map<int, memstoloc_t> init_tensor_to_memnode;

    map<string, scalar_t> vars;

    void train(int epochs);

    void load_tensors(map<string, dbuffer_t> weights);

    void load_tensor(int gid, dbuffer_t d);


    private:
        graph_setup_t make_graph();
         
};