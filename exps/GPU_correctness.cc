#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/communicator.h"
#include "../src/engine/gpu/workspace.h"
#include "../src/server/gpu/server.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/engine/communicator.h"
#include "../src/autoplace/apart.h"
#include "../src/autoplace/alocate.h"
#include "../src/einsummable/gwriter.h"

#include "../src/server/base.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#include "../src/base/setup.h"
#include "gpu_kernel_manager.h"
#include "utility.h"
#include "../src/einsummable/reference.h"

#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <tuple>
#include <vector>

// print the information of the memgraph
void print_memgraph(memgraph_t memgraph)
{
    // print the input and output of every node
    for (int i = 0; i < memgraph.nodes.size(); ++i) {
        auto node = memgraph.nodes[i];
        // print device location
        std::cout << "Device: " << node.op.get_loc() << " ";
        std::cout << "Node " << i << " has input: ";
        for (auto in : node.inns) {
            std::cout << in << " ";
        }
        std::cout << "and output: ";
        for (auto out : node.outs) {
            std::cout << out << " ";
        }
        std::cout << "Node type: ";
        node.op.print_type();
        if (node.op.is_touch()) {
            // print the group id
            std::cout << " Group id: " << node.op.get_apply().group;
            auto mem_touch = node.op.get_apply().mems[0];
            // print the touch size
            std::cout << " Touch size: " << mem_touch.size;
        }
        if (node.op.is_move()) {
            // print src and dst device
            std::cout << " Src device: " << node.op.get_move().get_src_loc();
            std::cout << " Dst device: " << node.op.get_move().get_dst_loc();
        }
        std::cout << std::endl;
    }
}

auto taskgraph_stats(taskgraph_t taskgraph)
{
    int      num_input_msgs = 0;
    uint64_t num_input_bytes = 0;
    int      num_core_msgs = 0;
    uint64_t num_core_bytes = 0;
    set<int> inputs_everywhere = taskgraph.get_input_everywhere_ids();
    for (int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
        auto const& node = taskgraph.nodes[tid];
        if (node.op.is_move()) {
            uint64_t sz = node.op.get_move().size;
            if (inputs_everywhere.count(tid) > 0) {
                num_input_msgs++;
                num_input_bytes += sz;
            } else {
                num_core_msgs++;
                num_core_bytes += sz;
            }
        }
    }

    auto to_mb = [](uint64_t n) { return double(n) / 1e6; };
    DOUT("Printing taskgraph stats");
    // DOUT("# nodes that are not moves: " << taskgraph.nodes.size() - num_input_msgs -
    // num_core_msgs);
    DOUT("input Moves: " << num_input_msgs << "#, " << to_mb(num_input_bytes) << "MB, "
                         << "Core Moves: " << num_core_msgs << "#, " << to_mb(num_core_bytes)
                         << "MB");
}

auto memgraph_mem_stats(memgraph_t memgraph)
{
    DOUT("Number of nodes in memgraph: " << memgraph.nodes.size());
    int      num_moves = 0;
    int      num_loads = 0;
    int      num_evicts = 0;
    uint64_t num_move_bytes = 0;
    uint64_t num_load_bytes = 0;
    uint64_t num_evict_bytes = 0;
    for (int tid = 0; tid != memgraph.nodes.size(); ++tid) {
        auto const& node = memgraph.nodes[tid];
        if (node.op.is_move()) {
            uint64_t sz = node.op.get_move().size;
            num_moves++;
            num_move_bytes += sz;
        } else if (node.op.is_evict()) {
            uint64_t sz = node.op.get_evict().src.size;
            num_evicts++;
            num_evict_bytes += sz;
        } else if (node.op.is_load()) {
            uint64_t sz = node.op.get_load().dst.size;
            num_loads++;
            num_load_bytes += sz;
        }
    }

    auto to_mb = [](uint64_t n) { return double(n) / 1e6; };
    DOUT("Printing memgraph stats");
    DOUT("Moves: " << num_moves << "#, " << to_mb(num_move_bytes) << "MB, "
                   << "Loads: " << num_loads << "#, " << to_mb(num_load_bytes) << "MB, "
                   << "Evicts: " << num_evicts << "#, " << to_mb(num_evict_bytes) << "MB");
}

// check if the offset in mems is greater than the bound
// throw an error if it is
// also check if the offset + size is greater than the bound
void mem_t_check(std::vector<mem_t> mems, uint64_t bound)
{
    for (auto mem : mems) {
        if (mem.offset > bound) {
            // print the offset and bound
            std::cout << "Offset: " << mem.offset << " Bound: " << bound << std::endl;
            throw std::runtime_error("Error: offset is greater than the bound.");
        }
        if (mem.offset + mem.size > bound) {
            // print the offset and bound
            std::cout << "Offset: " << mem.offset << " Bound: " << bound << std::endl;
            throw std::runtime_error("Error: offset + size is greater than the bound.");
        }
    }
}

// check if all nodes in the memgraph are within the memory bound
void check_bounds(memgraph_t memgraph, uint64_t bound)
{
    // print the bound
    for (auto node : memgraph.nodes) {
        if (node.op.is_inputmem()) {
            auto op = node.op.get_inputmem();
            if (op.offset + op.size > bound) {
                std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: INPUTMEM");
            }
        } else if (node.op.is_apply()) {
            auto op = node.op.get_apply();
            mem_t_check(op.mems, bound);
        } else if (node.op.is_move()) {
            auto op = node.op.get_move();
            if (std::get<1>(op.src) + op.size > bound) {
                std::cout << "Offset: " << std::get<1>(op.src) << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: MOVE");
            }
            if (std::get<1>(op.dst) + op.size > bound) {
                std::cout << "Offset: " << std::get<1>(op.dst) << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: MOVE");
            }
        } else if (node.op.is_evict()) {
            memloc_t src = node.op.get_evict().src;
            if (src.offset + src.size > bound) {
                std::cout << "Offset: " << src.offset << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: EVICT");
            }
        } else if (node.op.is_load()) {
            memloc_t dst = node.op.get_load().dst;
            if (dst.offset + dst.size > bound) {
                std::cout << "Offset: " << dst.offset << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: LOAD");
            }
        } else if (node.op.is_partialize()) {
            auto op = node.op.get_partialize();
            if (op.offset + op.size > bound) {
                std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: PARTIALIZE");
            }
        } else if (node.op.is_alloc()) {
            auto op = node.op.get_alloc();
            if (op.offset + op.size > bound) {
                std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: ALLOC");
            }
        } else if (node.op.is_del()) {
            auto op = node.op.get_del();
            if (op.offset + op.size > bound) {
                std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
                throw std::runtime_error("Memory bound exceeded: DEL");
            }
        }
    }
}

tuple<graph_t, vector<partition_t>> build_matmul_even_splits(uint64_t n, int n_split)
{
    partdim_t   pd = partdim_t::split(n, n_split);
    partition_t part2({pd, pd});
    partition_t part3({pd, pd, pd});

    graph_constructor_t g;
    int                 lhs = g.insert_input(part2);
    int                 rhs = g.insert_input(part2);
    int join = g.insert_einsummable(part3, einsummable_t::from_matmul(n, n, n), {lhs, rhs});
    int out = g.insert_formation(part2, join, true);

    auto const&         graph = g.graph;
    vector<partition_t> parts = vector_from_each_member(g.get_placements(), partition_t, partition);

    return {graph, parts};
}

vector<placement_t> autoplace(graph_t const& graph, int num_gpus, int num_parts_per_gpu = 1)
{
    int max_branching = 1;

    auto parts = apart01(graph, num_gpus * num_parts_per_gpu, max_branching);

    uint64_t flops_per_byte_moved = 100;

    return alocate01(graph, parts, num_gpus, flops_per_byte_moved);
}

void translate_execute(memgraph_t memgraph, bool debug, int num_gpus_per_node)
{
    if (debug) {
        print_memgraph(memgraph);
    }

    DOUT("Translate and execute memgraph");

    auto num_gpu = memgraph.mem_sizes().size();
    // allocate ptrs for gpu
    std::vector<void*> gpu_ptrs;
    auto               mem_sizes = memgraph.mem_sizes();
    for (int i = 0; i < num_gpu; ++i) {
        gpu_ptrs.push_back(gpu_allocate_memory(mem_sizes[i], i));
        // print mem_sizes
        std::cout << "mem_sizes[" << i << "]: " << mem_sizes[i] << std::endl;
    }

    vector<kernel_manager_t> kms;
    for (int i = 0; i < num_gpu; ++i) {
        kms.push_back(kernel_manager_t(i));
    }

    exec_graph_t graph =
        exec_graph_t::make_gpu_exec_graph(memgraph, 0, kms, num_gpus_per_node, gpu_ptrs);

    streampool_t stream_pool;
    stream_pool.initialize(5, 4);

    gpu_storage_t storage;

    rm_ptr_t resource_manager(
        new resource_manager_t(vector<rm_ptr_t>{rm_ptr_t(new gpu_workspace_manager_t()),
                                                rm_ptr_t(new group_manager_t()),
                                                rm_ptr_t(new global_buffers_t(gpu_ptrs)),
                                                rm_ptr_t(new gpu_storage_manager_t(&storage)),
                                                rm_ptr_t(new streampool_manager_t(stream_pool))}));

    exec_state_t state(graph, resource_manager);

    DOUT("executing...");
    state.event_loop();
    DOUT("executed.");
}

tuple<graph_t, vector<placement_t>>
build_graph_pls(int world_size, uint64_t matrix_dim, int partition)
{
    uint64_t ni, nj, nk;
    int      li, lj;
    int      rj, rk;
    int      ji, jj, jk;
    int      oi, ok;
    ni = nj = nk = matrix_dim;
    li = lj = partition;
    rj = rk = partition;
    ji = jj = jk = partition;
    oi = ok = partition;

    graph_constructor_t g;
    dtype_t             dtype = default_dtype();

    int lhs_1 = g.insert_input(partition_t({partdim_t::split(ni, li), partdim_t::split(nj, lj)}));
    int rhs_1 = g.insert_input(partition_t({partdim_t::split(nj, rj), partdim_t::split(nk, rk)}));

    int join_1 = g.insert_einsummable(
        partition_t({partdim_t::split(ni, ji), partdim_t::split(nk, jk), partdim_t::split(nj, jj)}),
        einsummable_t::from_matmul(ni, nj, nk),
        {lhs_1, rhs_1});

    int out_1 = g.insert_formation(
        partition_t({partdim_t::split(ni, oi), partdim_t::split(nk, ok)}), join_1);

    int lhs_2 = g.insert_input(partition_t({partdim_t::split(ni, li), partdim_t::split(nj, lj)}));
    int rhs_2 = g.insert_input(partition_t({partdim_t::split(nj, rj), partdim_t::split(nk, rk)}));

    int join_2 = g.insert_einsummable(
        partition_t({partdim_t::split(ni, ji), partdim_t::split(nk, jk), partdim_t::split(nj, jj)}),
        einsummable_t::from_matmul(ni, nj, nk),
        {lhs_2, rhs_2});

    int out_2 = g.insert_formation(
        partition_t({partdim_t::split(ni, oi), partdim_t::split(nk, ok)}), join_2);

    int join_out = g.insert_einsummable(
        partition_t({partdim_t::split(ni, ji), partdim_t::split(nk, jk), partdim_t::split(nj, jj)}),
        einsummable_t::from_matmul(ni, nj, nk),
        {out_1, out_2});

    int final_out = g.insert_formation(
        partition_t({partdim_t::split(ni, oi), partdim_t::split(nk, ok)}), join_out);

    auto pls = g.get_placements();
    for (int i = 0; i != pls.size(); ++i) {
        DOUT(i << " " << pls[i].partition);
    }

    // randomly assign the locations
    if (world_size > 1) {
        for (auto& pl : pls) {
            for (auto& loc : pl.locations.get()) {
                loc = runif(world_size);
            }
        }
    }

    return {g.graph, pls};
}

void server_execute_mm(int world_size, uint64_t matrix_dim, int partition)
{
    communicator_t c("0.0.0.0", true, world_size);

    // create a map for local insert tensors
    map<int, tuple<int, buffer_t>> data;
    uint64_t                       mem_size = 6lu * 1024lu * 1024lu * 1024lu;
    vector<uint64_t>               buffer_sizes;
    for (int i = 0; i < world_size; ++i) {
        buffer_sizes.push_back(mem_size);
    }

    gpu_mg_server_t server(c, buffer_sizes);
    server.set_split_off_inputs(true);

    auto [graph, pls] = build_graph_pls(world_size, matrix_dim, partition);

    // initialize input tensors and distribute across the cluster
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t   tensor = make_dbuffer(input.dtype, product(input.shape));
            // tensor.random("-0.01", "0.01");
            tensor.ones();
            DOUT("Printing input tensor...");
            DOUT(tensor);
            server.insert_tensor(gid, pls[gid], tensor);
        }
    }
    // DOUT("Printing graphviz...")
    // std::ofstream f("g_multiply.gv");
    // graph.print_graphviz(f);

    server.execute_graph(graph, pls);

    //// get the outputs to here
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_save()) {
            dbuffer_t tensor = server.get_tensor_from_gid(gid);
            DOUT("Printing output tensor...");
            DOUT(tensor);
            // DOUT("gid sum is: " << tensor.sum());
        }
    }

    server.shutdown();
}

void server_execute_multiple_mm(int world_size, uint64_t matrix_dim, int num_gpus)
{
    communicator_t c("0.0.0.0", true, world_size);

    // create a map for local insert tensors
    map<int, tuple<int, buffer_t>> data;
    uint64_t                       mem_size = 14lu * 1024lu * 1024lu * 1024lu;
    vector<uint64_t>               buffer_sizes;
    for (int i = 0; i < num_gpus; ++i) {
        buffer_sizes.push_back(mem_size);
    }

    graph_writer_t g;
    auto           A = g.input({matrix_dim, matrix_dim});
    auto           B = g.input({matrix_dim, matrix_dim});
    // auto C = g.input({matrix_dim, matrix_dim});
    // auto D = g.input({matrix_dim, matrix_dim});
    auto E = g.matmul(A, B).save();
    // auto F = g.matmul(C,D).save();
    // auto G = g.matmul(E,F).save();
    graph_t graph = g.get_graph();

    auto pls = autoplace(graph, num_gpus);

    auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

    bool use_storage = true;
    bool split_off_inputs = true;

    auto [_2, _3, maybe_init_memgraph, core_memgraph] =
        memgraph_t::make_(taskgraph,
                          {},
                          buffer_sizes,
                          {},
                          allocator_settings_t::gpu_alignment_settings(),
                          use_storage,
                          split_off_inputs);

    std::cout << "mm_basic.gv" << std::endl;
    std::ofstream f("mm_basic.gv");
    core_memgraph.print_graphviz(f);

    gpu_mg_server_t server(c, buffer_sizes);
    server.set_split_off_inputs(true);

    // initialize input tensors and distribute across the cluster
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t   tensor = make_dbuffer(input.dtype, product(input.shape));
            tensor.random("-0.01", "0.01");
            // tensor.ones();
            // DOUT("Printing input tensor...");
            // DOUT(tensor);
            server.insert_tensor(gid, pls[gid], tensor);
        }
    }

    server.execute_graph(graph, pls);

    //// get the outputs to here
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_save()) {
            dbuffer_t tensor = server.get_tensor_from_gid(gid);
            // DOUT("Printing output tensor...");
            // DOUT(tensor);
            // DOUT("gid sum is: " << tensor.sum());
        }
    }

    server.shutdown();
}

void server_execute_mm_partition(uint64_t matrix_dim, int num_gpus, int partition)
{
    communicator_t c("0.0.0.0", true, 1);

    // create a map for local insert tensors
    map<int, tuple<int, buffer_t>> data;
    uint64_t                       mem_size = 14lu * 1024lu * 1024lu * 1024lu;
    vector<uint64_t>               buffer_sizes;
    for (int i = 0; i < num_gpus; ++i) {
        buffer_sizes.push_back(mem_size);
    }

    DOUT("num_gpus: " << num_gpus);
    DOUT("partition: " << partition);
    auto [graph, part] = build_matmul_even_splits(matrix_dim, partition);
    auto pls = alocate01(graph, part, num_gpus, 100);

    auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

    taskgraph_stats(taskgraph);

    bool use_storage = true;
    bool split_off_inputs = true;

    auto [_2, _3, maybe_init_memgraph, core_memgraph] =
        memgraph_t::make_(taskgraph,
                          {},
                          buffer_sizes,
                          {},
                          allocator_settings_t::gpu_alignment_settings(),
                          use_storage,
                          split_off_inputs);

    std::cout << "mm_partition.gv" << std::endl;
    std::ofstream f("mm_partition.gv");
    core_memgraph.print_graphviz(f);

    gpu_mg_server_t server(c, buffer_sizes);
    server.set_split_off_inputs(true);

    // initialize input tensors and distribute across the cluster
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t   tensor = make_dbuffer(input.dtype, product(input.shape));
            tensor.random("-0.01", "0.01");
            // tensor.ones();
            // DOUT("Printing input tensor...");
            // DOUT(tensor);
            server.insert_tensor(gid, pls[gid], tensor);
        }
    }

    server.execute_graph(graph, pls);

    //// get the outputs to here
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_save()) {
            dbuffer_t tensor = server.get_tensor_from_gid(gid);
            // DOUT("Printing output tensor...");
            // DOUT(tensor);
            // DOUT("gid sum is: " << tensor.sum());
        }
    }

    server.shutdown();
}

// void contractionTest(int di, int dj, int dk) {
//   auto num_elems = di * dj + dj * dk + di * dk;
//   auto buffer_size = num_elems * sizeof(float);
//   // create the einsummable
//   auto einsummable = einsummable_t::from_matmul(di, dj, dk);
//   einsummable = einsummable.merge_adjacent_dims();
//   // create two input dbuffers
//   dbuffer_t input1 = make_dbuffer(dtype_t::f32, di * dj);
//   dbuffer_t input2 = make_dbuffer(dtype_t::f32, dj * dk);
//   // input1.random("-1.0", "1.0");
//   // input2.random("-1.0", "1.0");
//   input1.ones();
//   input2.ones();
//   dbuffer_t output = make_dbuffer(dtype_t::f32, di * dk);
//   output.random("-1.0", "1.0");
//   // print cpu output
//   std::cout << "CPU output: " << std::endl;
//   printFloatCPU(reinterpret_cast<const float*>(output.data->data), di * dk);

//   auto km = kernel_manager_t();
//   auto maybe_built = km.build(einsummable);
//   if (!maybe_built) {
//     throw std::runtime_error("Failed to build einsummable");
//   }
//   auto built = maybe_built.value();
//   if (!built.known()){
//     throw std::runtime_error("Workspace is unknown");
//   }
//   auto workspace_size = km.workspace_size(einsummable).value();

//   auto device = 0;

//   cudaSetDevice(device);

//   dbuffer_t cpu_out = reference_einsummable(einsummable, {input1, input2});

//   auto gpu_input1 = gpu_allocate_memory(input1.data->size, device);
//   auto gpu_input2 = gpu_allocate_memory(input2.data->size, device);
//   auto gpu_output = gpu_allocate_memory(cpu_out.data->size, device);

//   cuda_stream_t stream;
//   if (cudaMemcpy(gpu_input1, input1.data->data, input1.data->size,
//                  cudaMemcpyHostToDevice) != cudaSuccess) {
//     throw std::runtime_error("cudaMemcpy input 1");
//   }
//   if (cudaMemcpy(gpu_input2, input2.data->data, input2.data->size,
//                  cudaMemcpyHostToDevice) != cudaSuccess) {
//     throw std::runtime_error("cudaMemcpy input 2");
//   }
//   if (cudaMemcpy(gpu_output, output.data->data, output.data->size,
//                  cudaMemcpyHostToDevice) != cudaSuccess) {
//     throw std::runtime_error("cudaMemcpy output");
//   }

//   if (workspace_size > 0){
//     auto workspace_ptr = gpu_allocate_memory(workspace_size, device);
//     km(einsummable, stream.stream, gpu_output, {gpu_input1, gpu_input2},
//      std::make_tuple(workspace_ptr, workspace_size));
//   }
//   else{
//     km(einsummable, stream.stream, gpu_output, {gpu_input1, gpu_input2});
//   }
//   cudaDeviceSynchronize();

//   std::cout << "GPU input 1: " << std::endl;
//   printFloatGPU(reinterpret_cast<const float*>(gpu_input1), di * dj);
//   std::cout << "GPU input 2: " << std::endl;
//   printFloatGPU(reinterpret_cast<const float*>(gpu_input2), dj * dk);
//   std::cout << "GPU output: " << std::endl;
//   printFloatGPU(reinterpret_cast<const float*>(gpu_output), di * dk);

//   // dbuffer_t gpu_out = make_dbuffer(
//   //     dtype_t::f32, std::floor(cpu_out.data->size / sizeof(float)));
//   // if (cudaMemcpy(gpu_out.data->data, gpu_output, cpu_out.data->size,
//   //                cudaMemcpyDeviceToHost) != cudaSuccess) {
//   //   throw std::runtime_error("cudaMemcpy");
//   // }

//   // // compare the results
//   // auto result = is_close(cpu_out, gpu_out);
//   // // print messages based on the result
//   // if (result) {
//   //   std::cout << "Contraction test passed" << std::endl;
//   // } else {
//   //   std::cout << "Contraction test failed" << std::endl;
//   // }

//   // if (!result) {
//   //   std::cout << "Expected result: " << std::endl;
//   //   printFloatCPU(reinterpret_cast<const float *>(cpu_out.data->data),
//   //                 std::floor(cpu_out.data->size / sizeof(float)));
//   //   std::cout << "Actual result: " << std::endl;
//   //   printFloatCPU(reinterpret_cast<const float *>(gpu_out.data->data),
//   //                 std::floor(gpu_out.data->size / sizeof(float)));
//   // }
// }

graph_t generate_ffnn(uint64_t batch, vector<uint64_t> dims)
{
    graph_writer_t writer;

    using tensor_t = graph_writer_t::tensor_t;

    tensor_t x = writer.input({batch, dims[0]});

    tensor_t out = x;

    for (auto dim = 1; dim < dims.size(); ++dim) {
        out = writer.matmul(out, writer.input({dims[dim - 1], dims[dim]}));
        printf("out shape: %lu %lu\n", dims[dim - 1], dims[dim]);
        if (dim != dims.size() - 1) {
            out = writer.ew(scalarop_t::make_relu(), out);
            printf("Relu\n");
        } else {
            printf("Softmax\n");
            out = writer.softmax_v1(out);
        }
    }
    out.save_inplace();

    auto graph = writer.get_graph();

    // print the graph
    DOUT("Printing graphviz for ffnn...");
    std::ofstream f("ffnn.gv");
    graph.print_graphviz(f);

    return graph;
}

// SOFTMAX(RELU(X * W_1) * W_2)
graph_t ffnn_specific()
{
    graph_writer_t writer;

    using tensor_t = graph_writer_t::tensor_t;

    uint64_t batch_size = 256;
    // H_1 and H_2 are different hidden dimensions of the Hidden layer (1 hidden layer only)
    uint64_t H_1 = 1 << 10;
    uint64_t H_2 = 1 << 14;
    uint64_t output_class = 1 << 14;
    uint64_t input_dim = 1 << 19;

    tensor_t x = writer.input({batch_size, input_dim});
    tensor_t out = x;

    // [batch_size, input_dim] * [input_dim, H_1] = [batch_size, H_1]
    out = writer.matmul(out, writer.input({input_dim, H_2}));
    printf("MATMUL Matrix 1: %lu %lu\n", batch_size, input_dim);
    printf("MATMUL Matrix 2: %lu %lu\n", input_dim, H_2);
    out = writer.ew(scalarop_t::make_relu(), out);
    printf("Relu\n");
    // [batch_size, H_1] * [H_2, output_class] = [batch_size, output_class]
    out = writer.matmul(out, writer.input({H_2, output_class}));
    printf("MATMUL Matrix 1: %lu %lu\n", batch_size, H_2);
    printf("MATMUL Matrix 2: %lu %lu\n", H_2, output_class);
    out = writer.softmax_v1(out);
    printf("Softmax\n");

    out.save_inplace();

    return writer.get_graph();
}

// SOFTMAX(RELU(X * W_1) * W_2)
graph_t ffnn_specific_H1()
{
    graph_writer_t writer;

    using tensor_t = graph_writer_t::tensor_t;

    uint64_t batch_size = 256;
    // H_1 and H_2 are different hidden dimensions of the Hidden layer (1 hidden layer only)
    uint64_t H_1 = 1 << 10;
    uint64_t H_2 = 1 << 14;
    uint64_t output_class = 1 << 14;
    uint64_t input_dim = 1 << 19;

    tensor_t x = writer.input({batch_size, input_dim});
    tensor_t out = x;

    // [batch_size, input_dim] * [input_dim, H_1] = [batch_size, H_1]
    out = writer.matmul(out, writer.input({input_dim, H_1}));
    printf("MATMUL Matrix 1: %lu %lu\n", batch_size, input_dim);
    printf("MATMUL Matrix 2: %lu %lu\n", input_dim, H_1);
    out = writer.ew(scalarop_t::make_relu(), out);
    printf("Relu\n");
    // [batch_size, H_1] * [H_1, output_class] = [batch_size, output_class]
    out = writer.matmul(out, writer.input({H_1, output_class}));
    printf("MATMUL Matrix 1: %lu %lu\n", batch_size, H_1);
    printf("MATMUL Matrix 2: %lu %lu\n", H_1, output_class);
    out = writer.softmax_v1(out);
    printf("Softmax\n");

    out.save_inplace();

    return writer.get_graph();
}

void lowerTri_test()
{
    cudaSetDevice(0);
    kernel_manager_t km(0);

    dbuffer_t input1 = make_dbuffer(dtype_t::f32, 16);
    input1.ones();

    fill_t test = fill_t::make_square_lowertri(dtype_t::f32, 4);
    printf("Lower: %f, Upper: %f, Start: %lu\n",
           ((float*)test.get_lowertri().lower.raw())[0],
           ((float*)test.get_lowertri().upper.raw())[0],
           test.get_lowertri().start);

    auto          gpu_mem = gpu_allocate_memory(16 * sizeof(float), 0);
    cuda_stream_t stream = cuda_stream_t();
    cudaMemcpy(gpu_mem, input1.data->data, 16 * sizeof(float), cudaMemcpyHostToDevice);
    printFloatGPU(gpu_mem, 16);
    km.lowerTri_fill(test.get_lowertri(), stream.stream, gpu_mem);
    cudaDeviceSynchronize();

    printFloatGPU(gpu_mem, 16);
}

void constant_test()
{
    cudaSetDevice(0);
    kernel_manager_t km(0);

    dbuffer_t input1 = make_dbuffer(dtype_t::f32, 16);
    input1.zeros();

    auto my_scalar = scalar_t::one(dtype_t::f32);

    fill_t test = fill_t::make_constant(my_scalar, {16});
    printf("Constant: %f\n", ((float*)test.get_constant().value.raw())[0]);

    auto          gpu_mem = gpu_allocate_memory(16 * sizeof(float), 0);
    cuda_stream_t stream = cuda_stream_t();
    cudaMemcpy(gpu_mem, input1.data->data, 16 * sizeof(float), cudaMemcpyHostToDevice);
    printFloatGPU(gpu_mem, 16);
    km.constant_fill(test.get_constant(), stream.stream, gpu_mem);

    auto cpu_ptr = malloc(16 * sizeof(float));

    cudaMemcpy(cpu_ptr, gpu_mem, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printFloatGPU(gpu_mem, 16);
    // printFloatCPU((float*)cpu_ptr, 16);
}

void ew_test()
{
    // writing an einsum, running it on the GPU and comparing the results
    // create the scalarop for the einsum; we are testing elementwise
    // identity
    dtype_t    type = dtype_t::f32;
    scalarop_t arg1 = scalarop_t::make_arg(0, type);
    scalarop_t arg2 = scalarop_t::make_arg(1, type);
    scalarop_t add = scalarop_t::make_add(type);
    scalarop_t mul = scalarop_t::make_mul(type);
    scalarop_t power = scalarop_t::make_power(2, type);
    scalarop_t exp = scalarop_t::make_exp(type);
    scalarop_t ew = scalarop_t::combine(exp, {arg1});
    // print the scalarop
    DOUT("Testing scalarop: " << ew);

    // create the einsummable
    // einsummable_t(
    // vector<uint64_t> join_shape,
    // vector<vector<int>> inns,
    // int out_rank,
    // scalarop_t join,
    // optional<castable_t> castable = std::nullopt);
    vector<uint64_t>    join_shape = {16};
    vector<vector<int>> inns = {{0}};
    int                 out_rank = 1;
    einsummable_t       einsum = einsummable_t(join_shape, inns, out_rank, ew);
    auto                num_elems = product(join_shape);

    // create cpu buffers
    dbuffer_t input1 = make_dbuffer(dtype_t::f32, num_elems);
    dbuffer_t input2 = make_dbuffer(dtype_t::f32, num_elems);
    input1.random("-1.0", "1.0");
    input2.random("-1.0", "1.0");
    dbuffer_t output = make_dbuffer(dtype_t::f32, num_elems);
    // run cpu reference on this einsum
    auto cpu_out = reference_einsummable(einsum, {input1});
    // create gpu buffers
    auto gpu_input1 = gpu_allocate_memory(input1.data->size, 0);
    auto gpu_input2 = gpu_allocate_memory(input2.data->size, 0);
    auto gpu_output = gpu_allocate_memory(output.data->size, 0);
    // copy inputs to gpu
    cudaMemcpy(gpu_input1, input1.data->data, input1.data->size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_input2, input2.data->data, input2.data->size, cudaMemcpyHostToDevice);

    // create the kernel manager
    kernel_manager_t km(0);
    // create the gpu stream
    cuda_stream_t stream = cuda_stream_t();
    // build the elementwise
    auto maybe_built = km.build(einsum);
    if (!maybe_built) {
        throw std::runtime_error("Failed to build ew einsummable");
    }
    // get the workspace size
    auto built = maybe_built.value();
    if (!built.known()) {
        throw std::runtime_error("Workspace is unknown for elemmentwise");
    }
    auto workspace_size = built.value();
    // allocate workspace
    if (workspace_size != 0) {
        DOUT("NOTE: using workspace for elementwise");
    }
    auto                workspace_ptr = gpu_allocate_memory(workspace_size, 0);
    vector<void const*> inputs = {gpu_input1};
    // run the elementwise
    km(einsum, stream.stream, gpu_output, inputs, std::make_tuple(workspace_ptr, workspace_size));

    // bring back the output to the cpu
    cudaMemcpy(output.data->data, gpu_output, output.data->size, cudaMemcpyDeviceToHost);

    // compare the results
    auto result = is_close(cpu_out, output);
    if (result) {
        std::cout << "Elementwise test passed" << std::endl;
    } else {
        std::cout << "Elementwise test failed" << std::endl;
        // print the cpu and gpu outputs
        std::cout << "CPU output: " << std::endl;
        printFloatCPU(reinterpret_cast<const float*>(cpu_out.data->data),
                      std::floor(cpu_out.data->size / sizeof(float)));
        std::cout << "GPU output: " << std::endl;
        printFloatCPU(reinterpret_cast<const float*>(output.data->data),
                      std::floor(output.data->size / sizeof(float)));
    }
}
