#include "exec_nodes.h"
#include "../exec_graph.h"
#include "managers.h"
#include "workspace.h"
#include "storage.h"
#include "stream_pool.h"
#include "utility.h"
#include <iostream>
#include <sys/types.h>

void print_exec_graph(exec_graph_t exec_graph){
  for (int i = 0; i < exec_graph.nodes.size(); ++i) {
    auto node = exec_graph.nodes[i];
    std::cout << "Node " << i << " has input: ";
    for (auto in : node.inns) {
      std::cout << in << " ";
    }
    std::cout << "and output: ";
    for (auto out : node.outs) {
      std::cout << out << " ";
    }
    std::cout << std::endl;
  }
}

// TODO: have a stream pool implementation once creating a stream on fly works
exec_graph_t exec_graph_t::make_gpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  kernel_manager_t& gpu_km,
  int num_gpus_per_node,
  vector<void*> gpu_mems)
{
  exec_graph_t graph;

  map<int, int> mid_to_eid;

  auto insert = [&](op_ptr_t op, int mid)
  {
    auto const& node = memgraph.nodes[mid];
    auto const& mid_inns = node.inns;
    auto const& mid_outs = node.outs;

    vector<int> inns;
    for(auto const& mid: mid_inns) {
      inns.push_back(mid_to_eid.at(mid));
    }

    int eid = graph.insert(op, inns);

    mid_to_eid.insert({mid, eid});
  };

  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    auto const& node = memgraph.nodes[mid];
    if(!node.op.is_local_to_gpu(this_rank, num_gpus_per_node)) {
      DOUT("Skipping node " << mid << " because it is not local to this gpu")
      continue;
    }
    // DOUT("Making exec graph for node " << mid);

    if(
      node.op.is_inputmem()   ||
      node.op.is_inputsto()   ||
      node.op.is_partialize() ||
      node.op.is_alloc()      ||
      node.op.is_del())
    {
      op_ptr_t op = std::make_shared<dummy_t>();
      insert(op, mid);
      // DOUT("Inserted dummy op for node " << mid);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      if(apply.is_einsummable()) {
        if(apply.group >= 0) {
          throw std::runtime_error("only allowing touches to have a group");
        }

        auto einsum = apply.get_einsummable();
        auto einsum_merged = apply.get_einsummable().merge_adjacent_dims();
        // build the op (except the workspace size)
        gpu_einsummable_t* op = new gpu_einsummable_t(
          gpu_km,
          einsum,
          apply.mems,
          node.op.get_apply_loc()
        );

        // compile the kernel (and update the workspace size)
        auto maybe_built = gpu_km.build(op->einsummable);
        if(!maybe_built) {
          throw std::runtime_error("GPU KM could not compile the kernel");
        }
        auto workspace_info = maybe_built.value();
        if (workspace_info.known()){
          op->workspace_size = workspace_info.value();
        }
        else{
          int loc = node.op.get_apply_loc();
          // get the input and output memory ptrs
          void* out_mem = increment_void_ptr(
            gpu_mems[loc],
            apply.mems[0].offset);
          vector<void const*> inn_mems;
          inn_mems.reserve(apply.mems.size() - 1);
          for(int i = 1; i != apply.mems.size(); ++i) {
            inn_mems.push_back(increment_void_ptr(
              gpu_mems[loc],
              apply.mems[i].offset));
          }
          // get the workspace size
          op->workspace_size = gpu_km.known_workspace_size(einsum_merged, out_mem, inn_mems);
        }
        // insert into the graph
        insert(op_ptr_t(op), mid);
        // DOUT("Inserted einsummable op for node " << mid);
      } else if(apply.is_touch()) {
        gpu_touch_t* op = new gpu_touch_t(
          gpu_km,
          apply.get_touch(),
          apply.group,
          apply.mems,
          node.op.get_apply_loc()
        );

        // Any touch that does not have a group id is the only write to
        // the output bytes, so make sure it's castable is none so that
        // it does a copy and not a sum
        if(op->group_id < 0) {
          op->touch.castable = std::nullopt;
        }

        insert(op_ptr_t(op), mid);
        // DOUT("Inserted touch op for node " << mid);
      } else {
        throw std::runtime_error("should not reach");
      }
    } else if(node.op.is_move()) {
      // check if the move is local to this gpu
      auto const& move = node.op.get_move();
      auto const& src = move.get_src_loc();
      auto const& dst = move.get_dst_loc();
      if (std::floor(src/num_gpus_per_node) != std::floor(dst/num_gpus_per_node)) {
        throw std::runtime_error("node to node communication not supported yet");
      }
      gpu_copy_t* op = new gpu_copy_t(node.op.get_move());
      insert(op_ptr_t(op), mid);
      // DOUT("Inserted copy op for node " << mid);
    } else if(node.op.is_evict()) {
      throw std::runtime_error("GPU evict not implemented");
    } else if(node.op.is_load()) {
      throw std::runtime_error("GPU load not implemented");
    } else {
      throw std::runtime_error("should not reach");
    }
  }
  // Debug: print the exec_graph
  // print_exec_graph(graph);
  return graph;
}

desc_ptr_t
gpu_einsummable_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  // 3rd: a workspace (if workspace_size > 0)
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));

  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  if (workspace_size > 0) {
    gpu_workspace_desc_t workspace_desc;
    workspace_desc.device = device;
    workspace_desc.size = workspace_size;
    ret.emplace_back(gpu_workspace_manager_t::make_desc(workspace_desc));
  }
  return resource_manager_t::make_desc(ret);
}

void gpu_einsummable_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  DOUT("gpu_einsummable_t::launch: getting resources")
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);
  DOUT("number of resources: " << resources.size());

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* out_mem = increment_void_ptr(
    global_buffer,
    mems[0].offset);

  vector<void const*> inn_mems;
  inn_mems.reserve(mems.size() - 1);
  for(int i = 1; i != mems.size(); ++i) {
    inn_mems.push_back(increment_void_ptr(
      global_buffer,
      mems[i].offset));
  }
  optional<tuple<void*, uint64_t>> maybe_workspace;
  if(workspace_size > 0) {
    maybe_workspace = gpu_workspace_manager_t::get_resource(resources[2]).as_tuple();
  }

  cudaStream_t stream = streampool_manager_t::get_resource(resources[1]).stream;
  // create stream and launch
  // cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  
  gpu_km(
    einsummable,
    stream,
    out_mem,
    inn_mems,
    maybe_workspace);

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  DOUT("gpu_einsummable_t::launch: adding callback");

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_einsummable_t: callback");
}

desc_ptr_t
gpu_touch_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  // 3rd: a group id (if group_id >= 0)
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc());

  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  if(group_id >= 0) {
    ret.emplace_back(group_manager_t::make_desc(group_id));
  }

  return resource_manager_t::make_desc(ret);
}

void gpu_touch_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* out_mem = increment_void_ptr(
    global_buffer,
    mems[0].offset);

  void const* inn_mem = increment_void_ptr(
    global_buffer,
    mems[1].offset);

  bool is_first = false;
  if(group_id >= 0) {
    tuple<int, bool> const& info = group_manager_t::get_resource(resources[2]);
    is_first = std::get<1>(info);
  }

  touch_t this_touch = touch;
  if(is_first) {
    // if this is the first touch, make sure the touch becomes a copy
    this_touch.castable = std::nullopt;
  }

  // create stream and launch
  cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;
  gpu_km(
    this_touch,
    stream,
    out_mem,
    inn_mem);

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_touch_t: callback");
}

desc_ptr_t
gpu_copy_t::resource_description() const
{
  // 1st: gpu memory ptrs (this could be one resource or two depending on the design)
  // 2nd: a stream
  vector<desc_ptr_t> ret;
  auto my_move = move;
  auto [src_loc, src_offset] = my_move.src;
  auto [dst_loc, dst_offset] = my_move.dst;
  ret.emplace_back(global_buffers_t::make_desc(src_loc));
  ret.emplace_back(global_buffers_t::make_desc(dst_loc));
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{dst_loc}));

  return resource_manager_t::make_desc(ret);
}

void gpu_copy_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto src_buffer = global_buffers_t::get_resource(resources[0]);
  auto dst_buffer = global_buffers_t::get_resource(resources[1]);

  auto [src_loc, src_offset] = move.src;
  auto [dst_loc, dst_offset] = move.dst;

  void* src_mem = increment_void_ptr(
    src_buffer,
    src_offset);

  void* dst_mem = increment_void_ptr(
    dst_buffer,
    dst_offset);

  cudaSetDevice(src_loc);
  // auto stream = streampool_manager_t::get_resource(resources[2]).stream;
  cudaStream_t stream = cuda_create_stream();
  cudaError_t cudaError = cudaMemcpyAsync(dst_mem, src_mem, move.size, cudaMemcpyDeviceToDevice, stream);
  if (cudaError != cudaSuccess) {
    // print the error code and error string
    fprintf(stderr, "cudaMemcpy failed with error: %s\n", cudaGetErrorString(cudaError));
    // print src and dst loc
    fprintf(stdout, "src_loc: %d\n", src_loc);
    fprintf(stdout, "dst_loc: %d\n", dst_loc);
    // print src and dst buffer
    fprintf(stdout, "src_buffer: %p\n", src_buffer);
    fprintf(stdout, "dst_buffer: %p\n", dst_buffer);
    // print src and dst mem
    fprintf(stdout, "src_mem: %p\n", src_mem);
    fprintf(stdout, "dst_mem: %p\n", dst_mem);
    // print cpy size
    fprintf(stderr, "cpy size: %zu\n", move.size);

    throw std::runtime_error("cudaMemcpy failed");
  }
}
