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

#include <iostream>
#include <fstream>


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

    int loss_stoid = -1;
    int predictions_stoid = -1;

    std::ofstream file;

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
    map<int, memstoloc_t> init_tensor_to_memstoloc;
    map<string, int> model_weight_map;

    map<string, scalar_t> vars;

    void train(int epochs, vector<vector<dbuffer_t>> batches, vector<vector<dbuffer_t>> labels);

    void load_tensors(map<string, dbuffer_t> weights);

    void load_tensor(int gid, dbuffer_t d);

    dbuffer_t get_tensor(string name);

    dbuffer_t get_tensor(int gid);


    private:
        graph_setup_t make_graph();
         
};