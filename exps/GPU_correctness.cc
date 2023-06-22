#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute.h"
#include "../src/einsummable/reference.h"

#include <fstream>
#include <memory>

void check_correctness(memgraph_t memgraph, bool debug = false){
    if (debug) {
         // print the number of nodes in the graph
        std::cout << "Number of nodes in the graph: " << memgraph.nodes.size() << std::endl;
        // print the input and output of every node
        for(int i = 0; i < memgraph.nodes.size(); ++i) {
        std::cout << "Node " << i << " has input: ";
        for(auto in: memgraph.nodes[i].inns) {
            std::cout << in << " ";
        }
        std::cout << "and output: ";
        for(auto out: memgraph.nodes[i].outs) {
            std::cout << out << " ";
        }
        std::cout << "Node type: ";
        memgraph.nodes[i].op.print_type();
        std::cout << std::endl;
        }
    }

    // print a message
    std::cout << "Checking correctness" << std::endl;
    // create a buffer
    auto num_elems =  memgraph.mem_sizes()[0] / sizeof(float);
    dbuffer_t d = make_dbuffer(dtype_t::f32, num_elems);
    // d.random("-1.0", "1.0");
    d.fill(scalar_t(float(1.0)));
    buffer_t b = d.data;
    auto cpu_ptr = b->data;
    auto size = b->size;
    // allocate a buffer on GPU
    auto gpu_ptr = gpu_allocate_memory(memgraph.mem_sizes()[0]);
    // copy data from CPU to GPU
    if(cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy");
    }
    // execute the memgraph on the GPU ptr
    execute(memgraph, gpu_ptr);
    std::cout << "GPU execution has finished" << std::endl;
    // bring the data back
    dbuffer_t out = make_dbuffer(dtype_t::f32, num_elems);
    if(cudaMemcpy(out.data->data, gpu_ptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy");
    }
    std::cout << "Copying from GPU to CPU has finished" << std::endl;

    // run the same computation on CPU
    vector<buffer_t> buffers;
    buffers.push_back(b);
    reference_compute_memgraph(memgraph, buffers);
    std::cout << "CPU reference has finished" << std::endl;

    // compare the results
    auto result = is_close(d, out);
    // print messages based on the result
    if(result) {
        std::cout << "Correctness test passed" << std::endl;
    } else {
        std::cout << "Correctness test failed" << std::endl;
    }

    if (debug && !result){
        std::cout << "Expected result: " << std::endl;
        printFloatCPU(reinterpret_cast<const float*>(cpu_ptr), num_elems);
        std::cout << "Actual result: " << std::endl;
        printFloatCPU(reinterpret_cast<const float*>(out.data->data), num_elems);
    }

}