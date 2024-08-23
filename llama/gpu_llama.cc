#include "gpu_llama.h"

gpu_llama_t::gpu_llama_t(uint64_t mem_size, uint64_t sto_size, llama_size_t llama_size) : storage_size(sto_size), llama_size(llama_size), communicator(communicator_t("0.0.0.0", true, 1)), graph(make_graph()) {

    file.open("lora_tensors.txt");

    vector<uint64_t> buffer_sizes(1, mem_size);

    // communicator = communicator_t("0.0.0.0", true, 1);
    int this_rank = communicator.get_this_rank();

    server = std::make_unique<gpu_mg_server_t>(communicator, buffer_sizes, sto_size);

    // Create checkpoint taskgraphs

    // Used to figure out required shapes
    auto const& margs = graph.margs;

    // Use graphs to get checkpoint_taskgraphs_t
    auto const& graphs = graph.checkpoint_graphs;

    // updater_desc will be used to set learning rate scalar variables
    auto const& updater_desc = graph.updater_desc;

    // these fills need to be added before executing
    auto const& init_fills = graph.init_fills;

    // used for the next remap
    auto const& old_news = graph.old_news;

    auto start_pls = std::chrono::high_resolution_clock::now();
    vector<placement_t> full_pls = autoplace01(graph.full_graph, autoplace_config_t::make_default01(
    1, 1));
    for (auto &[old_gid, new_gid] : graph.old_news) {
        full_pls[new_gid] = full_pls[old_gid];
    }
    taskgraphs = std::make_unique<checkpoint_taskgraphs_t>(graphs, full_pls);

    for (auto& [name, gid] : graph.model_weight_map) {
        model_weight_map[name] = gid;
    }

    auto end_pls = std::chrono::high_resolution_clock::now();
    DOUT("placement time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_pls - start_pls).count() << " ms");

    // Create checkpoint memgraphs
    auto start_mgmake = std::chrono::high_resolution_clock::now();

    // Start by constructing first memgraph
    auto &[init_rel, tg, save_rel] = taskgraphs->infos[0];
    vector<allocator_t> allocators;
    input_tid_to_datas.emplace_back();
    for (auto &msize : buffer_sizes) {
        allocators.emplace_back(msize, allocator_settings_t::gpu_alignment_settings());
    }
    states.emplace_back(std::make_unique<memgraph_make_state_t>(tg, vector<int>(buffer_sizes.size(), 0), allocators, input_tid_to_datas.back(), 1, 0, true));
    std::unique_ptr<memgraph_make_state_t>& state = states.back();

    for(int id = 0; id != tg.nodes.size(); ++id)
    {
        auto const& node = tg.nodes[id];
        if(node.op.is_input())
        {
            state->initialize_input(id);
        }
    }
    state->process(order_taskgraph(tg));
    // print_graphviz(state.memgraph, "llamamg0.gv");
    for (auto &[tid, node] : input_tid_to_datas.back()) {
        init_tensor_to_memstoloc[tid] = node;
    }

    for (int i = 1; i < taskgraphs->infos.size(); i++) {
        // DOUT("Creating memgraph " << i)
        auto &[_prev_init, _last_tg, save_rel] = taskgraphs->infos[i-1];
        auto &[init_rel, current_tg, _next_save] = taskgraphs->infos[i];
        std::unique_ptr<memgraph_make_state_t>& prev_state = states[i-1];
        input_tid_to_datas.emplace_back();
        map<int, memstoloc_t>& remappings = input_tid_to_datas.back();

        for (auto &[old_gid, new_gid] : graphs.remaps[i]) {
            auto& old_tids = save_rel.at(old_gid).tids.get();
            auto& new_tids = init_rel.at(new_gid).tids.get();
            for (int which_tid = 0; which_tid < save_rel.at(old_gid).tids.get().size(); which_tid++) {

                int old_tid = old_tids[which_tid];
                int new_tid = new_tids[which_tid];

                int old_mid =
                prev_state->task_tensor_to_mem_node[old_tid];
                auto& old_node = prev_state->memgraph.nodes[old_mid];
                memstoloc_t old_memstoloc = old_node.op.get_output_memstoloc();

                remappings[new_tid] = old_memstoloc;
            }
        }
        // DOUT("Building next memgraph");
        vector<allocator_t> allocators;
        for (auto &msize : buffer_sizes) {
            allocators.emplace_back(msize, allocator_settings_t::gpu_alignment_settings());
        }

        // DOUT("Initializing state");
        states.emplace_back(std::make_unique<memgraph_make_state_t>(current_tg,  vector<int>(buffer_sizes.size(), 0), allocators, remappings, 1, 0, true));
        std::unique_ptr<memgraph_make_state_t>& state = states.back();


        state->_sto_id = states[i-1]->_sto_id+1;
        // DOUT("Processing nodes");
        auto ordering = order_taskgraph(current_tg);
        state->process(ordering);
    }

    auto end_mgmake = std::chrono::high_resolution_clock::now();
    DOUT("Memgraph make time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_mgmake - start_mgmake).count() << " ms");

    // Extend final memgraph to reconnect to initial memgraph
    auto start_mgextend = std::chrono::high_resolution_clock::now();

    // Map gids that we want to move memory between to their respective gids in the first checkpoint graph
    // Final_gid_remaps is a map from a nodes gid in the overall graph to its gid in the final cg graph
    map<int, int> final_gid_remaps;
    for (auto &[old_gid, new_gid] : graphs.remaps[graphs.remaps.size()-1]) {
        final_gid_remaps[new_gid] = old_gid;
    }

    // initial_gid_remaps is a map from a nodes gid in the overall graph to its gid in the initial cg graph
    map<int, int> initial_gid_remaps;
    for (auto &[old_gid, new_gid] : graphs.remaps[0]) {
        initial_gid_remaps[old_gid] = new_gid;
    }

    // gid_remappings is a map from the gid of a node that we need to map back to the first graph in the final cg graph to 
    // the gid of the corresponding node in the first cg graph
    map<int, int> gid_remappings;
    for (auto &[old_gid, new_gid] : graph.old_news) {
        gid_remappings[final_gid_remaps[new_gid]] = initial_gid_remaps[old_gid];
    }

    auto &[initial_rel, __last_tg, __save_rel] = taskgraphs->infos[0];
    auto &[__init_rel, __current_tg, final_rel] = taskgraphs->infos.back();

    std::unique_ptr<memgraph_make_state_t>& init_state = states[0];
    std::unique_ptr<memgraph_make_state_t>& final_state = states.back();

    vector<tuple<memstoloc_t, memstoloc_t, int>> new_mem_mapping;

    for (auto &[final_cg_gid, initial_cg_gid] : gid_remappings) {
        auto &final_tids = final_rel.at(final_cg_gid).tids.get();
        auto &init_tids = initial_rel.at(initial_cg_gid).tids.get();

        if (final_tids.size() != init_tids.size()) {
            throw std::runtime_error("Size of relations do not match up, make sure corresponding gids have the same placements");
        }

        for (int which_tid = 0; which_tid < final_rel.at(final_cg_gid).tids.get().size(); which_tid++) {

            int final_tid = final_tids[which_tid];
            int init_tid = init_tids[which_tid];

            // int init_mid =
            // init_state.input_tid[init_tid];
            // auto &init_node = init_state.memgraph.nodes[init_mid];

            int final_mid = final_state->task_tensor_to_mem_node[final_tid];
            auto &final_node = final_state->memgraph.nodes[final_mid];

            memstoloc_t init_memstoloc = input_tid_to_datas[0][init_tid];
            // if (init_node.op.is_inputmem()) {
            //   init_memstoloc = memstoloc_t(init_node.op.get_inputmem().as_memloc());
            // } else if (init_node.op.is_inputsto()) {
            //   init_memstoloc = memstoloc_t(init_node.op.get_inputsto().as_stoloc());
            // } else {
            //   DOUT(init_node.op.get_name());
            //   throw std::runtime_error("Matching node is not inputmem or inputsto");  
            // }
            memstoloc_t final_memstoloc = final_node.op.get_output_memstoloc();
            // if (init_memstoloc.is_stoloc() && init_memstoloc.get_stoloc().id == 125) {
            //   DOUT("final mid: " << final_mid);
            //   DOUT(final_node.op.get_name());
            //   if (final_memstoloc.is_memloc()) {
            //     DOUT(final_memstoloc.get_memloc());
            //   } else {
            //     DOUT(final_memstoloc.get_stoloc());
            //   }
            // }

            new_mem_mapping.emplace_back(final_memstoloc, init_memstoloc, final_mid);
        }
    }

    

    // Move loss, predictions, to new stoids to keep track
    int loss_cg_gid = final_gid_remaps[loss_id];
    int loss_tid = final_rel.at(loss_cg_gid).tids.get()[0];
    int loss_mid = final_state->task_tensor_to_mem_node[loss_tid];
    auto &loss_node = final_state->memgraph.nodes[loss_mid];
    memstoloc_t loss_memstoloc = loss_node.op.get_output_memstoloc();
    memstoloc_t loss_new_memstoloc = memstoloc_t(stoloc_t{.loc = 0, .id = ++final_state->_sto_id});
    loss_stoid = final_state->_sto_id;

    new_mem_mapping.emplace_back(loss_memstoloc, loss_new_memstoloc, loss_mid);

    // DOUT("Through loss");

    DOUT("Predictions id " << predictions_id);
    DOUT("Loss id" << loss_id);
 
    int predictions_cg_gid = final_gid_remaps.at(predictions_id);
    DOUT("Predictions cg gid" << predictions_cg_gid);
    int predictions_tid = final_rel.at(predictions_cg_gid).tids.get()[0];
    int predictions_mid = final_state->task_tensor_to_mem_node[predictions_tid];
    auto &predictions_node = final_state->memgraph.nodes[predictions_mid];
    memstoloc_t predictions_memstoloc = predictions_node.op.get_output_memstoloc();
    memstoloc_t predictions_new_memstoloc = memstoloc_t(stoloc_t{.loc = 0, .id = ++final_state->_sto_id});
    predictions_stoid = final_state->_sto_id;

    // DOUT("Through loss & pred");

    // new_mem_mapping.emplace_back(predictions_memstoloc, predictions_new_memstoloc, predictions_mid);


    remaps.push_back(final_state->move_tensors(new_mem_mapping));

    auto end_mgextend = std::chrono::high_resolution_clock::now();
    DOUT("Memgraph extend time: " << 
    std::chrono::duration_cast<std::chrono::milliseconds>(end_mgextend - start_mgextend).count() << " ms");


    for (auto &[from, to] : graph.checkpoint_graphs.remaps[0]) {
        gid_to_init_cg[from] = to;
    }
    map<string, int> model_weights;
    for (auto &[name, gid] : graph.model_weight_map) {
        model_weights[name] = gid;
    }

    scalar_t _lr(dtype, write_with_ss(learning_rate));

    vars["beta1"] = scalar_t(dtype, "0.9");
    vars["beta2"] = scalar_t(dtype, "0.999");
    vars["eta"] = _lr;
    vars["learning_rate"] = _lr;

    update_vars(updater_desc, 1, vars);
}

graph_setup_t gpu_llama_t::make_graph() {

    if(seed >= 0) {
        set_seed(seed);
    }

    {
        int n_layers = max_n_layers;
        if(n_layers >= 0) {
            margs.n_layers = std::min(margs.n_layers, n_layers);
        }
    }

    margs.max_seq_len = seq_len;

    graph_t graph;
    vector<int> checkpoints;
    vector<int> weight_ids;
    vector<int> grad_ids;
    set<int> forward_ids;

    vector<int> constant_ids;
    vector<tuple<string, int>> model_weight_map;
    {
        graph_writer_t writer;
        transformer_t model(&writer, margs, 0, lora_rank);

        tensor_t embeddings = writer.input(full_shape_t({
            full_dim_t::singleton(margs.batch_size),
            full_dim_t::singleton(margs.max_seq_len),
            margs.full_dim()
        }));

        // predictions: batch size, vocab size
        tensor_t predictions = model.forward(embeddings);
        predictions.save_inplace();
        tensor_t labels = writer.input(
        vector<uint64_t>{margs.batch_size, margs.vocab_size},
        dtype);

        // Compute the loss
        //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
        //   Loss = sum_n (l{n}) / N
        // Note, shift by c for numerical stability;
        //   where c{n} = max_c v{n,c}
        tensor_t loss;
        {
            tensor_t v = predictions;
            tensor_t c = writer.reduction("bv->b", castable_t::max, v);
            // v = v - c
            v = writer.ew("bv,b->bv", scalarop_t::make_sub(dtype), v, c);
            // ev = exp(v)
            tensor_t ev = writer.ew(scalarop_t::make_exp(dtype), v);
            // evsubset{b} = sum_v ev{b,v}*labels{b,v}
            tensor_t evsubset = writer.contraction("bv,bv->b", ev, labels);
            tensor_t evsum    = writer.reduction("bv->b", castable_t::add, ev);

            tensor_t lll = writer.ew(
                "b,b->b",
                scalarop_t::make_div(dtype),
                evsubset, evsum);

            lll = writer.ew(scalarop_t::make_log(dtype), lll);

            // (would like to use unsqueeze here but it is not implemented)

            double one_over_bsz = 1.0 / double(margs.batch_size);
            loss = lll.scale(scalar_t(dtype, write_with_ss(one_over_bsz)));
        }
        loss.save_inplace();

        vector<tensor_t> ws;
        vector<tensor_t> cs;
        for(auto [name, tensor]: model.weight_map()) {
            model_weight_map.emplace_back(name, tensor.get_id());

            if(name.find("lora") != string::npos) {
                ws.push_back(tensor);
            } else {
                if(lora_rank) {
                    // we're doing lora, so explicitly save all the weights
                    tensor.save_inplace();
                    // and add them as a constant
                    cs.push_back(tensor);
                }
            }
        }

        {
            vector<int> fs = vector_iota<int>(writer.get_graph().nodes.size());
            forward_ids = set<int>(fs.begin(), fs.end());
        }

        vector<tensor_t> grads = writer.backprop(loss, ws);

        checkpoints = vector_from_each_method(model.checkpoints, int, get_id);
        weight_ids  = vector_from_each_method(ws, int, get_id);
        grad_ids    = vector_from_each_method(grads, int, get_id);

        embeddings_id = embeddings.get_id();
        predictions_id = predictions.get_id();
        labels_id = labels.get_id();
        loss_id = loss.get_id();

        full_freqs_cis_id = model.full_freqs_cis.get_id();
        model.full_freqs_cis.save_inplace();

        constant_ids = vector_from_each_method(cs, int, get_id);

        graph = std::move(writer.get_graph());
    }

    updater_desc_t::vanilla_t vanilla {};
    updater_desc_t::adamw_t adamw { .min_precision = dtype_t::f32 };
    updater_desc = is_adamw ?
        updater_desc_t { dtype, adamw   }    :
        updater_desc_t { dtype, vanilla }    ;

    vector<tuple<int, int>> old_news;
    for(auto const& constant_id: constant_ids) {
        old_news.emplace_back(constant_id, constant_id);
    }

    old_news.emplace_back(full_freqs_cis_id, full_freqs_cis_id);

    vector<tuple<int, fill_t>> init_fills = update_weights(
        updater_desc, graph, old_news, weight_ids, grad_ids);

    // Note that update_weights may add input nodes, which must belong
    // to the forward graph
    for(auto const& [new_input_id, _]: init_fills) {
        forward_ids.insert(new_input_id);
    }

    // Make sure that every new is a save
    for(auto const& [old_id, new_id]: old_news) {
        graph.nodes[new_id].op.set_save(true);
    }

    checkpoint_graphs_t checkpoint_graphs(
        graph,
        checkpoints,
        forward_ids);

    // DOUT("Vocab sign " << margs.vocab_size);
    // DOUT("Batch Size " << margs.batch_size);
    // DOUT("Max Seq Len" << margs.max_seq_len); 

    // std::cout << "Embeddings shape: ("; 
    // for (auto& dim : graph.out_shape(embeddings_id)) {
    //     std::cout << dim << ", ";
    // }
    // std::cout << std::endl;

    return graph_setup_t {
        .margs = margs,
        .full_graph = graph,
        .checkpoint_graphs = checkpoint_graphs,
        .updater_desc = updater_desc,
        .old_news = old_news,
        .init_fills = init_fills,
        .embeddings_id = embeddings_id,
        .predictions_id = predictions_id,
        .labels_id = labels_id,
        .loss_id = loss_id,
        .full_freqs_cis_id = full_freqs_cis_id,
        .model_weight_map = model_weight_map
    };
}

void gpu_llama_t::train(int epochs, vector<vector<dbuffer_t>> batches, vector<vector<dbuffer_t>> labels) {
    int embeddings_shape = product(graph.full_graph.out_shape(embeddings_id));

    if (batches.size() != labels.size()) {
        throw std::runtime_error("Wrong number of inputs or labels");
    }

    file << get_tensor("layers.0.attention.lora0.wk.weight") << "\n\n";
    file << get_tensor("layers.0.attention.lora1.wk.weight") << "\n\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        DOUT("Epoch " << epoch);
        // Execute all memgraphs
        for (int i = 0; i < batches.size(); i++) {
            auto& batch = batches[i];
            auto& label = labels[i];
            for (int j = 0; j < batch.size(); j++) {
                auto& masked_tokens = batch[i];
                auto& one_hot_label = label[i];

                if (masked_tokens.dtype == dtype && masked_tokens.nelem() == embeddings_shape && one_hot_label.nelem() == margs.batch_size * margs.vocab_size) {
                    load_tensor(embeddings_id, masked_tokens);
                    load_tensor(labels_id, one_hot_label);

                    for (int i = 0; i < states.size(); i++) {
                        server->execute_memgraph(states[i]->memgraph, false, vars);
                    }
                    server->storage_remap_server(remaps);

                    dbuffer_t preds = dbuffer_t(dtype, server->get_storage_buf(predictions_stoid));
                    DOUT("Preds" << preds);

                    dbuffer_t loss = dbuffer_t(dtype, server->get_storage_buf(loss_stoid));
                    DOUT("Loss = " << loss);
                } else {
                    if (!(masked_tokens.dtype == dtype && masked_tokens.nelem() == embeddings_shape)) {
                        DOUT("Skipping batch, size incorrect: " << masked_tokens.nelem() << " instead of " << embeddings_shape);
                    } else {
                        DOUT("Skipping batch, labels incorrect, have " << one_hot_label.nelem() << " instead of " << margs.batch_size * margs.vocab_size);
                    }
                    
                }
            }

        }
    }

    file << get_tensor("layers.0.attention.lora0.wk.weight") << "\n\n";
    file << get_tensor("layers.0.attention.lora1.wk.weight") << "\n\n";
}


void gpu_llama_t::load_tensor(int gid, dbuffer_t d) {
    if (gid_to_init_cg.find(gid) == gid_to_init_cg.end()) {
        throw std::runtime_error("Unavailable gid entered to get_tensor");
    }
    int cg_gid = gid_to_init_cg[gid];
    relation_t& new_rel = taskgraphs->infos[0].init_rel.at(cg_gid);

    relation_t rel = new_rel.as_singleton(0);
    remap_relations_t remap;
    remap.insert(rel, new_rel);

    map<int, buffer_t> data;
    data[0] = d.data;
    repartition(server->null_comm, remap, data, nullptr);

    for (auto &[tid, buf] : data) {
        
        if (init_tensor_to_memstoloc.find(tid) == init_tensor_to_memstoloc.end()) {
            throw std::runtime_error("Tid " + std::to_string(tid) + " does not exist in input_tid_to_data");
        } 

        memstoloc_t inn = init_tensor_to_memstoloc[tid];

        if (inn.is_memloc()) {
            auto& input = inn.get_memloc();

            // DOUT("Loading memory tensor with memloc offset " << input.offset);

            cudaError_t error = cudaMemcpy(increment_void_ptr(server->mems[input.loc], input.offset), buf->raw(), input.size, cudaMemcpyDefault);
            if(error != cudaSuccess) {
                DOUT(cudaGetErrorString(error));
                throw std::runtime_error("cudaMemcpy failed");
            }
        } else {
            auto& input = inn.get_stoloc();
            // DOUT("Loading storage tensor with stoid " << input.id);
            server->set_storage_buf(buf, input.id);
        }
    }
}

void gpu_llama_t::load_tensors(map<string, dbuffer_t> weights) {
    set<string> seen;
    for(auto const& [name, id]: graph.model_weight_map) {
        auto shape = graph.get_shape(id);
        // DOUT("\n")
        // DOUT("Loading tensor " << name << " with id " << id);
        if (weights.find(name) != weights.end()) {
            // DOUT("Loading tensor " << name << " from input tensors from pointer " << weights[name].raw());
            if (weights[name].nelem() != product(shape)) {
                throw std::runtime_error("Provided tensor for weight " + name + " does not have the correct shape, product of " + std::to_string(weights[name].nelem())  + " instead of " + std::to_string(product(shape)));
            }
            load_tensor(id, weights[name]);
            seen.insert(name);
        } else {
            if (name.find("lora") != string::npos) {
                // For the lora, we have (X*L0)*L1 where L0 needs to be
                // initialized gaussiann and L1 needs to be initialized with zeros
                dbuffer_t dbuffer = make_dbuffer(dtype, product(shape));

                if(name.find("lora0") != string::npos) {
                    dbuffer.rnorm();
                    dbuffer.scale(scalar_t(dtype, write_with_ss(float(1e-3))));
                } else if(name.find("lora1") != string::npos) {
                    dbuffer.zeros();
                } else {
                    throw std::runtime_error("should not reach");
                }
                // DOUT("Loading tensor " << id);
                load_tensor(id, dbuffer);
            } else {
                DOUT("Need to load tensor " << id);
                dbuffer_t dbuffer = make_dbuffer(dtype, product(shape));
                dbuffer.zeros();
                load_tensor(id, dbuffer);
                // int next_tid = server->get_max_tid() + 1;
                // relation_t relation = model_loader(
                //   register_cmd, name, shape, next_tid);
                // server->insert_gid_without_data(id, relation);
            }
        }
    }

    for (auto const& [name, dbuf] : weights) {
        if (seen.find(name) == seen.end()) {
            DOUT("Warning: did not load " << name);
        }
    }
}


dbuffer_t gpu_llama_t::get_tensor(int gid) {
    // DOUT("Getting tensor with gid " << gid);
    if (gid == loss_id) {
        return dbuffer_t(dtype, server->get_storage_buf(loss_stoid));
    }
    if (gid == predictions_id) {
        // TODO figure out why saving predictions throws error
        throw std::runtime_error("Not implemented");
    }
    if (gid_to_init_cg.find(gid) == gid_to_init_cg.end()) {
        throw std::runtime_error("Unavailable gid entered to get_tensor");
    }
    int cg_gid = gid_to_init_cg[gid];
    // DOUT("Initial checkpoint graph cggid " << cg_gid);
    relation_t& rel = taskgraphs->infos[0].init_rel.at(cg_gid);

    vector<uint64_t> shape = graph.full_graph.out_shape(gid);
    dtype_t dtype = graph.full_graph.out_dtype(gid);
    // relation_t rel = relation_t::make_singleton(dtype, shape, 0);
    // relation_t new_rel = relation_t::make_singleton(dtype, shape, 0);
    relation_t new_rel = rel.as_singleton(0);

    remap_relations_t remap;
    remap.insert(rel, new_rel);

    map<int, buffer_t> data;
    for (auto &tid : rel.tids.get()) {
        // DOUT("Rel tid " << tid);
        memstoloc_t input_memstoloc = init_tensor_to_memstoloc.at(tid);
        buffer_t out;

        if (input_memstoloc.is_memloc()) { // cudaMemcpy
            memloc_t input = input_memstoloc.get_memloc();
            out = make_buffer(input.size);

            cudaError_t error = cudaMemcpy(out->raw(), increment_void_ptr(server->mems[input.loc], input.offset), input.size, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                DOUT(cudaGetErrorString(error));
                throw std::runtime_error("cudaMemcpy failed");
            }
        } else { // storage write
            stoloc_t input = input_memstoloc.get_stoloc();
            out = server->get_storage_buf(input.id);
        }


        // int mid = states[0]->task_tensor_to_mem_node[tid];

        // auto &node = states[0]->memgraph.nodes[mid];

        // buffer_t out;

        // if (!node.op.is_inputmem() && !node.op.is_inputsto()) {
        //     throw std::runtime_error("Tensor to get does not correspond to inputmem or inputsto node");
        // }
        // if (node.op.is_inputmem()) {
        //     auto& input = node.op.get_inputmem();
        //     out = make_buffer(input.size);

        //     cudaError_t error = cudaMemcpy(out->raw(), increment_void_ptr(server->mems[input.loc], input.offset), input.size, cudaMemcpyDeviceToHost);
        //     if(error != cudaSuccess) {
        //         DOUT(cudaGetErrorString(error));
        //         throw std::runtime_error("cudaMemcpy failed");
        //     }
        // } else {
        //     auto& input = node.op.get_inputsto();
        //     out = server->get_storage_buf(input.storage_id);
        // }

        data[tid] = out;
    }

    repartition(server->null_comm, remap, data, nullptr);
    return dbuffer_t(dtype, data[0]);
}

dbuffer_t gpu_llama_t::get_tensor(string weight) {
    if (model_weight_map.find(weight) == model_weight_map.end()) {
        throw std::runtime_error("Requested weight does not exist");
    }
    return get_tensor(model_weight_map[weight]);
}