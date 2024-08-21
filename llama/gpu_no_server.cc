#include "gpu_llama.h"
// #include "misc.h"
// #include "modules.h"
// #include "reader.h"
// #include "dataset_reader.h"

// #include "../src/misc/checkpoint.h"
// #include "../src/misc/update.h"

communicator_t gpu_mg_server_t::null_comm;

int main() {
    gpu_llama_t model(125lu * 100lu * 1000lu * 1000lu, 10e10, llama_size_t::B7);

    // for (int i = 0; i < model.states.size(); i++) {
    //     DOUT("Memgraph " << i << " has " << model.states[i]->memgraph.nodes.size() << " nodes");
    // }

    map<string, dbuffer_t> weights;
    model.load_tensors(weights);

    for (auto& [name, dbuf] : model.model_weight_map) {
        model.get_tensor(name);
    }

    // model.train(1);
}