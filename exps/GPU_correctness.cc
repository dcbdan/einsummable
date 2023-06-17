#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute.h"
#include "../src/einsummable/reference.h"

#include <fstream>
#include <memory>

void check_correctness(memgraph_t memgraph, int num_elems){
    // print a message
    std::cout << "Checking correctness" << std::endl;
  // create a buffer
    dbuffer_t d = make_dbuffer(dtype_t::f32, num_elems);
    d.random("-1.0", "1.0");
    buffer_t b = d.data;
    auto cpu_ptr = b->data;
    auto size = b->size;
    // allocate a buffer on GPU
    auto gpu_ptr = gpu_allocate_memory(num_elems * sizeof(float));
    // copy data from CPU to GPU
    if(cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy");
    }
    // execute the memgraph on the GPU ptr
    execute(memgraph, gpu_ptr);
    // bring the data back
    dbuffer_t out = make_dbuffer(dtype_t::f32, num_elems);
    if(cudaMemcpy(out.data->data, gpu_ptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy");
    }

    // run the same computation on CPU
    printFloatCPU(reinterpret_cast<const float*>(cpu_ptr), num_elems);
    vector<buffer_t> buffers;
    buffers.push_back(b);
    reference_compute_memgraph(memgraph, buffers);

    // compare the results
    auto result = is_close(d, out);
    // print messages based on the result
    if(result) {
        std::cout << "Correctness test passed" << std::endl;
    } else {
        std::cout << "Correctness test failed" << std::endl;
    }

    printFloatCPU(reinterpret_cast<const float*>(cpu_ptr), num_elems);
    printFloatCPU(reinterpret_cast<const float*>(out.data->data), num_elems);
}