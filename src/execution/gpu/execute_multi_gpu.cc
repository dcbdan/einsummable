#include "execute_multi_gpu.h"
#include "utility.h"
#include <cuda_runtime_api.h>


bool multi_gpu_execute_state_t::has_stream() {
  for (auto &stream : stream_pool) {
    if (stream.size() != 0) {
      return true;
    }
  }
  return false;
}

vector<int> multi_gpu_execute_state_t::node_update(int node_idx) {
  auto const& node = memgraph.nodes[node_idx];

  DOUT("node update on " << node_idx);

  // TODO: hard coded index 0 since we only have 1 device
  // print a update message
  // printf("Node %d finished execution\n", node_idx);
  --num_nodes_remaining;
  vector<int> ready_nodes;
  for (auto out : node.outs) {
    dependency_count[out] -= 1;
    // print the node that got decremented
    // printf("Node %d has dependencies decreased\n", out);
    if (dependency_count[out] == 0) {
      ready_nodes.push_back(out);
      // std::cout << "Adding node " << out << " to ready nodes" << std::endl;
    }
  }
  // if this node is a touch, we find if there are any other nodes
  // in the same group that are waiting for this touch to finish
  if (node.op.is_touch()) {
    auto node_loc = node.op.get_apply_loc();
    // at this point we know that the it's not the first time we see this
    // touch's group id
    auto group_id = node.op.get_apply().group;
    if(group_id >= 0) {
      // remove the group id from the executing list since we are done
      // the touch should be executed at the apply location since touch is an apply
      auto& is_executing = group_id_executing[node_loc];
      auto iter = is_executing.find(group_id);
      if (iter == is_executing.end()) {
        throw std::runtime_error("Group id " + std::to_string(group_id) +
                                 " not found in executing list");
      }
      is_executing.erase(iter);

      // find if there are any other nodes in the same group that are waiting for
      // this touch to finish
      auto& waiting_idxs = groupID_to_nodeIDX[node_loc][group_id];
      if(waiting_idxs.size() != 0) {
        // get one of them and add it to the ready nodes
        auto touch_node_idx = waiting_idxs.front();
        waiting_idxs.pop();
        ready_nodes.push_back(touch_node_idx);
      }
    }
  }
  return ready_nodes;
}

void multi_gpu_execute_state_t::printContractionInfo(int node_idx, int num_elems) {
  auto node = memgraph.nodes[node_idx];
  if (!node.op.is_contraction()){
    throw std::runtime_error("Error: printContractionInfo called on a non-contraction node");
  }
  auto node_loc = node.op.get_apply_loc();
  auto memory_vector = node.op.get_apply().mems;
  // print offsets
  std::cout << "Offset 1: " << memory_vector[1].offset << std::endl;
  std::cout << "Offset 2: " << memory_vector[2].offset << std::endl;
  // print inputs
  auto input1 = offset_increment(memory_base_ptrs[node_loc], memory_vector[1].offset);
  auto input2 = offset_increment(memory_base_ptrs[node_loc], memory_vector[2].offset);
  auto output = offset_increment(memory_base_ptrs[node_loc], memory_vector[0].offset);

  std::cout << "Input 1: ";
  printFloatGPU(static_cast<float*>(input1), num_elems);
  std::cout << "Input 2: ";
  printFloatGPU(static_cast<float*>(input2), num_elems);
  std::cout << "Output: ";
  printFloatGPU(static_cast<float*>(output), num_elems);
}

void multi_gpu_execute_state_t::checkContractionOffset(int node_idx) {
  auto node = memgraph.nodes[node_idx];
  auto memory_vector = node.op.get_apply().mems;
  auto output_offset = memory_vector[0].offset;
  auto input1_offset = memory_vector[1].offset;
  auto input2_offset = memory_vector[2].offset;
  // check if the offsets are divisible by 4
  if (output_offset % 4 != 0 || input1_offset % 4 != 0 ||
      input2_offset % 4 != 0) {
    // print the offsets
    std::cout << "Offset 1: " << input1_offset << std::endl;
    std::cout << "Offset 2: " << input2_offset << std::endl;
    std::cout << "Offset 3: " << output_offset << std::endl;
    throw std::runtime_error("Offset is not divisible by 4");
  }
}

// add a vector of nodes to the queue
void multi_gpu_execute_state_t::add_to_pending_queue(std::vector<int> &nodes) {
  for (auto node : nodes) {
    pending_queue.push(node);
  }
  // print how many elements are in the queue
  // std::cout << "Queue size: " << queue.size() << std::endl;
}

// get the dependency count of each node
std::map<int, int> multi_gpu_execute_state_t::get_dependencies() {
  std::map<int, int> dependency_count;
  for (int i = 0; i < memgraph.nodes.size(); i++) {
    dependency_count[i] = memgraph.nodes[i].inns.size();
  }
  return dependency_count;
}

// check if all the nodes finished executing
bool multi_gpu_execute_state_t::is_complete() {
  auto idx = 0;
  if (num_nodes_remaining != 0) {
    return false;
  }

  // double check if we really finished everything and the counters are updated
  // correctly
  idx = 0;
  for (auto it = dependency_count.begin(); it != dependency_count.end(); it++) {
    if (it->second != 0) {
      throw std::runtime_error("Error: All nodes finished execution but the "
                               "dependency count doesn't match.");
    }
    idx++;
  }

  if (pending_queue.size() != 0) {
    throw std::runtime_error("Error: All nodes finished execution but there "
                             "are still nodes in the queue.");
  }
  return true;
}

// helper function to get the input memory pointers from a vector of mem_t
// input memory pointers are mem[1:]
std::vector<void const *> 
multi_gpu_execute_state_t::get_input_mem_ptrs(
  std::vector<mem_t> mem,
  void *memory_base_ptr) 
{
  std::vector<void const *> ret;
  for (int i = 1; i < mem.size(); i++) {
    ret.push_back(offset_increment(memory_base_ptr, mem[i].offset));
  }
  return ret;
}

multi_gpu_execute_state_t::multi_gpu_execute_state_t(
  memgraph_t const& input_memgraph, std::vector<void*> mem_ptr)
    : memgraph(input_memgraph), memory_base_ptrs(mem_ptr) 
{
  int num_gpus = memory_base_ptrs.size();
  int num_streams = 5;

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    cudaSetDevice(gpu);  // Set the current GPU device
      
    // create stream pools
    std::queue<cudaStream_t> my_stream_pool;
    for (int i = 0; i < num_streams; i++) {
      my_stream_pool.push(cuda_create_stream());
    }
    stream_pool.push_back(my_stream_pool);

    group_id_executing.push_back(std::set<int>());
    all_group_ids.push_back(std::set<int>());
    groupID_to_nodeIDX.push_back(std::map<int, std::queue<int>>());
  }
  dependency_count = get_dependencies();
  // get the number of nodes remaining for each device
  num_nodes_remaining = memgraph.nodes.size();

  // check all elements from the memgraph and add the nodes with no
  // dependencies to the apply_queue
  for (int i = 0; i < memgraph.nodes.size(); i++) {
    if (memgraph.nodes[i].inns.size() == 0) {
      pending_queue.push(i);
    }
    // check if the node is an einsummable
    if (memgraph.nodes[i].op.is_einsummable()) {
      // get the einsummable object
      auto my_einsummable = memgraph.nodes[i].op.get_einsummable();

      auto iter = einsum_worksizes.find(my_einsummable);
      if(iter == einsum_worksizes.end()) {
        auto maybe_worksize = km.build(my_einsummable);
        if(!maybe_worksize) {
          throw std::runtime_error("could not build " + write_with_ss(my_einsummable));
        }
        einsum_worksizes.insert({my_einsummable, maybe_worksize.value()});
      }
    }
  }
}

void execute_multi_gpu(const memgraph_t &memgraph, std::vector<void*> mem_ptrs) {
  DOUT("");
  DOUT("");
  DLINEOUT("execute_multi_gpu");
  // create a multi_gpu_execute_state_t
  multi_gpu_execute_state_t gpu_execute_state(memgraph, mem_ptrs);
  // gpu_execute_state.run();
  gpu_execute_state.run_create_stream();
}

// Callback struct that updates execution when a node (with a stream attached)
// finishes execution
// NOTE: now every node has a stream attached must be a apply / move node
struct callback_data_t {
  std::mutex *m_ptr;
  std::condition_variable *cv_ptr;
  multi_gpu_execute_state_t *my_state;
  int node_idx;
  bool debug = true;
  cudaStream_t stream;

  void operator()() {
    std::mutex &m = *m_ptr;
    auto &cv = *cv_ptr;
    {
      std::unique_lock lk(m);
      auto node_op = my_state->memgraph.nodes[node_idx].op;
      int node_loc = 0;
      if (node_op.is_apply()){
        node_loc = node_op.get_apply_loc();
      }
      else if (node_op.is_move()){
        auto move_op = node_op.get_move();
        auto [src_loc, src_offset] = move_op.src;
        node_loc = src_loc;
      }
      else{
        throw std::runtime_error("Error: Callback called on a non-apply/move node");
      }

      // update the queues since this node is finished
      my_state->finished_queue.push(node_idx);
      my_state->finished_streams[stream] = node_loc;
      //if (debug){
      //  printf("Callback: Node %d finished execution.\n", node_idx);
      //}
    }
    cv.notify_all();
  }

  callback_data_t (std::mutex *m_ptr, std::condition_variable *cv_ptr, 
    multi_gpu_execute_state_t *my_state, int node_idx, cudaStream_t stream) {
    this->m_ptr = m_ptr;
    this->cv_ptr = cv_ptr;
    this->my_state = my_state;
    this->node_idx = node_idx;
    this->stream = stream;
  }

  callback_data_t() {}
};

// ---------------- Calling streams from a stream pool version ----------------

// ------------------------ No stream pool version ------------------------

void multi_gpu_execute_state_t::run_create_stream() {

  bool debug = true;
  
  while (true) {

    if (is_complete()) {
      break;
    }

    // locking the mutex until the queue has new things to execute
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&] {
        // wake up if there are possible updates or new nodes to run
        return finished_queue.size() > 0 || pending_queue.size() > 0;
      });
    }

    // execute things that are in the apply_queue until the queue is empty
    while (pending_queue.size() != 0) {
			std::unique_lock lk(m);
      // print out the pending queue
      // get the first element in the queue
      auto node_idx = pending_queue.front();
      auto node = memgraph.nodes[node_idx];
      // remove the first element from the queue
      pending_queue.pop();
      // print all node indices from the pending queue
      if (debug){
        std::queue<int> temp_queue = pending_queue;
        vector<int> items;
        while (temp_queue.size() != 0) {
          items.push_back(temp_queue.front());
          temp_queue.pop();
        }
        DOUT("pending_queue: " << items << ".. popping " << node_idx);
      }
			lk.unlock();
      // std::cout << "Executing node: " << node_idx << std::endl;
      // execute the node
      if (node.op.is_inputmem() || node.op.is_inputsto() || node.op.is_del() ||
          node.op.is_partialize() || node.op.is_alloc()) {
        // do nothing but update the memgraph execution since that node is
        // finished
        //std::unique_lock lk(m);
        //auto new_nodes = node_update(node_idx);
        //if (debug){
        //  DOUT("node " << node_idx << " finished... adding " << new_nodes);
        //  //printf("Node %d finished execution. New nodes added: ", node_idx);
        //  //for (auto n : new_nodes) {
        //  //  printf("%d ", n);
        //  //}
        //  //printf("\n");
        //}
        //add_to_pending_queue(new_nodes);
        finished_queue.push(node_idx);
      //} else if(node.op.is_touch()) {
      //  finished_queue.push(node_idx);
      } else if(node_idx == 11) {
        finished_queue.push(node_idx);
      } else if (node.op.is_apply()) {
        DLINEOUT("is apply for node " << node_idx);
        int node_loc = node.op.get_apply_loc();
        cudaSetDevice(node_loc); 
        cudaStream_t stream = cuda_create_stream();
        DLINE;

        auto memory_vector = node.op.get_apply().mems;

        void* out_mem = 
          offset_increment(memory_base_ptrs[node_loc], memory_vector[0].offset);
        vector<void const*> inn_mems =
          get_input_mem_ptrs(memory_vector, memory_base_ptrs[node_loc]);


        // CASE: TOUCH
        if (node.op.is_touch()) {
          // std::cout << "Got a touch node" << std::endl;
          // TODO: doing a lock; see where is the best place to put this lock
          auto touch = node.op.get_touch();
          auto group_id = node.op.get_apply().group;
          // if we have found this group id in the list, we can't execute until
          // the previous one is done
          if(group_id < 0) {
            if(touch.castable != std::nullopt) {
              throw std::runtime_error("all castables without group id should be nullopt");
            }
          } else if(group_id_executing[node_loc].count(group_id) != 0) {
            // we can't execute this node since some other node is executing
            // with the same group id add this node to the map
            groupID_to_nodeIDX[node_loc][group_id].push(node_idx);
            // skipping the callback since this node didn't execute
            continue;
          } else {
            // else we are free to run this
            // see this is the first time seeing this group id
            auto& group_ids = all_group_ids[node_loc];
            if(touch.castable == std::nullopt) {
              throw std::runtime_error("all castables with a group id must not be nullopt");
            }
            if(group_ids.count(group_id) == 0) {
              // set the castable to nullopt
              touch.castable = std::nullopt;
            } else {
              group_ids.insert(group_id);
            }

            // add this group id to the executing set
            group_id_executing[node_loc].insert(group_id);

            // use the operator of kernel_manager to run touch
            DLINEOUT("launching a touch");
            km(
              touch,
              stream,
              out_mem,
              inn_mems[0]);
          }
        } else {
          DLINE;
          auto my_einsummable = node.op.get_einsummable();
          if (my_einsummable.is_contraction()) {
            // do a check of the offsets
            checkContractionOffset(node_idx);
          }

          auto einsum_iter = einsum_worksizes.find(my_einsummable);
          if (einsum_iter == einsum_worksizes.end()) {
            throw std::runtime_error("Error: Einsummable not built");
          }
          auto const& workspace_info = einsum_iter->second;

          // TODO TODO: gpu allocated memory must be managed!

          optional<tuple<void*, uint64_t>> maybe_workspace;
          if(workspace_info.known()) {
            uint64_t const& workspace_size = workspace_info.value();
            if(workspace_size > 0) {
              DLINE;
              maybe_workspace = tuple<void*, uint64_t>(
                gpu_allocate_memory(workspace_size, node_loc),
                workspace_size);
            }
          } else {
            // we don't know the workspace size
            DLINE;
            // TODO
            uint64_t size = km.known_workspace_size(my_einsummable, out_mem, inn_mems);
            void* work = gpu_allocate_memory(size, node_loc);
            maybe_workspace = tuple<void*, uint64_t>(work, size);
          }

          //use kernel_manager operator to run einsum
          DLINEOUT("launching an einsummable");
          km(
            my_einsummable,
            stream,
            out_mem,
            inn_mems,
            maybe_workspace);
        }
        //if (debug){
        //  std::cout << "Node " << node_idx << " has been scheduled to a stream" << std::endl;
        //}
        // after execution, we attach the stream with a callback function
        // get all the metadata needed for the callback
        callback_data_t *data = new callback_data_t(&m, &cv, this, node_idx, stream);
        // add the callback
        cudaStreamAddCallback(
            stream,
            [](CUstream_st *, cudaError, void *raw_data) {
              callback_data_t *data = static_cast<callback_data_t *>(raw_data);
              callback_data_t &f = *data;
              f();
              delete data;
            },
            static_cast<void *>(data), 0);
      }
      else if (node.op.is_move()) {
        auto move_op = node.op.get_move();
        // get the src and dst information
        auto [src_loc, src_offset] = move_op.src;
        auto [dst_loc, dst_offset] = move_op.dst;
        // we should switch to the src location for memcpy
        cudaSetDevice(src_loc);
        // cudaStream_t stream = stream_pool[src_loc].front();
        // stream_pool[src_loc].pop();
        cudaStream_t stream = cuda_create_stream();
        // print dst_loc and dst_offset, src_loc and src_offset
        //if (debug){
        //  std::cout << "dst_loc: " << dst_loc << std::endl;
        //  std::cout << "dst_offset: " << dst_offset << std::endl;
        //  std::cout << "src_loc: " << src_loc << std::endl;
        //  std::cout << "src_offset: " << src_offset << std::endl;
        //}
        gpu_comm.send(offset_increment(memory_base_ptrs[dst_loc], dst_offset),
                      offset_increment(memory_base_ptrs[src_loc], src_offset),
                      move_op.size, stream);

        // add the callback
        callback_data_t *data = new callback_data_t(&m, &cv, this, node_idx, stream);
        cudaStreamAddCallback(
            stream,
            [](CUstream_st *, cudaError, void *raw_data) {
              callback_data_t *data = static_cast<callback_data_t *>(raw_data);
              callback_data_t &f = *data;
              f();
              delete data;
            },
            static_cast<void *>(data), 0);
      } 
      else {
        // print a message saying that the operation is not supported and this
        // operation's type
        throw std::runtime_error("Error: Operation not supported: Type is "
                                 "among the following - evict, load");
      }
    }

    std::unique_lock lk(m);
    while (finished_queue.size() != 0) {
      // get the node index
      int node_idx = finished_queue.front();
      // pop the node index
      finished_queue.pop();
      // update the queue since this node is finished
      auto new_nodes = node_update(node_idx);
      DOUT("adding " << new_nodes << " to pending");
      add_to_pending_queue(new_nodes);
      //if (debug){
      //  printf("Node %d finished execution. New nodes added: ", node_idx);
      //  for (auto n : new_nodes) {
      //    printf("%d ", n);
      //  }
      //  printf("\n");
      //}
    }
    lk.unlock();
  }
}
