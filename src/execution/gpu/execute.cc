#include "execute.h"
#include <mutex>

cudaStream_t cuda_create_stream() {
  cudaStream_t ret;
  if (cudaStreamCreate(&ret) != cudaSuccess) {
    // print error message and error code
    printf("cudaStreamCreate failed with error code %d\n", cudaGetLastError());
    throw std::runtime_error("cuda_create_stream");
  }
  return ret;
}

// increment the pointer by the byte offset
// ONLY USE IF THE UNIT OF OFFSET IS BYTE
float *offset_increment(const float *ptr, int offset) {
  return (float *)((char *)ptr + offset);
}

// USE THIS IF THE UNIT OF OFFSET IS FLOAT
float *float_increment(float *ptr, int offset) { return ptr + offset; }

// prints float starting from ptr with count number of elements
void printFloatCPU(const float *cpu_ptr, int count) {
  for (int i = 0; i < count; ++i) {
    printf("%.2f ", cpu_ptr[i]);
  }
  printf("\n");
}
void printFloatGPU(const float *gpu_ptr, int count) {
  float *cpu_ptr = (float *)malloc(count * sizeof(float));
  cudaMemcpy(cpu_ptr, gpu_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  printFloatCPU(cpu_ptr, count);
  free(cpu_ptr);
}

void checkAlignment(cutensorHandle_t *handle, float *ptr,
                    cutensorTensorDescriptor_t desc) {
  uint32_t alignmentRequirement;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, ptr, &desc,
                                               &alignmentRequirement));

  if (alignmentRequirement != 16) {
    // print the alignment requirement
    std::cout << "*** Alignment requirement mismatch; alignment: "
              << alignmentRequirement << std::endl;
  }
}

void init_value(float *ptr, int count, float value) {
  // malloc memory on cpu and cudamemcpy to gpu
  float *tmp = (float *)malloc(count * sizeof(float));
  float *check = (float *)malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i) {
    tmp[i] = value;
  }
  cudaMemcpy(ptr, tmp, count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(tmp, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(check, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
}

vector<int> gpu_execute_state_t::node_update(int node_idx) {
  // TODO: hard coded index 0 since we only have 1 device
  // print a update message
  // printf("Node %d finished execution\n", node_idx);
  num_nodes_remaining[0] -= 1;
  vector<int> ready_nodes;
  auto node = memgraph.nodes[node_idx];
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
    // at this point we know that the it's not the first time we see this
    // touch's group id
    auto group_id = node.op.get_apply().group;
    // remove the group id from the executing list since we are done
    if (group_id_executing.find(group_id) == group_id_executing.end()) {
      throw std::runtime_error("Group id " + std::to_string(group_id) +
                               " not found in executing list");
    }
    group_id_executing.erase(group_id);
    // find if there are any other nodes in the same group that are waiting for
    // this touch to finish
    if (groupID_to_nodeIDX[group_id].size() != 0) {
      // get one of them and add it to the ready nodes
      auto touch_node_idx = groupID_to_nodeIDX[group_id].front();
      groupID_to_nodeIDX[group_id].pop();
      // std::cout << "Adding touch node " << touch_node_idx << " to ready
      // nodes" << std::endl;
      ready_nodes.push_back(touch_node_idx);
    }
  }
  return ready_nodes;
}

void gpu_execute_state_t::printContractionInfo(int node_idx, int num_elems) {
	auto node = memgraph.nodes[node_idx];
  auto memory_vector = node.op.get_apply().mems;
  // print offsets
  std::cout << "Offset 1: " << memory_vector[1].offset << std::endl;
  std::cout << "Offset 2: " << memory_vector[2].offset << std::endl;
  // print inputs
  std::cout << "Input 1: ";
  printFloatGPU(offset_increment(memory_base_ptr, memory_vector[1].offset),
                num_elems);
  std::cout << "Input 2: ";
  printFloatGPU(offset_increment(memory_base_ptr, memory_vector[2].offset),
                num_elems);
  std::cout << "Output: ";
  printFloatGPU(offset_increment(memory_base_ptr, memory_vector[0].offset),
                num_elems);
}
void gpu_execute_state_t::checkContractionOffset(int node_idx) {
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
void gpu_execute_state_t::add_to_pending_queue(std::vector<int> &nodes) {
  for (auto node : nodes) {
    pending_queue.push(node);
  }
  // print how many elements are in the queue
  // std::cout << "Queue size: " << queue.size() << std::endl;
}

// get the dependency count of each node
std::map<int, int> gpu_execute_state_t::get_dependencies() {
  std::map<int, int> dependency_count;
  for (int i = 0; i < memgraph.nodes.size(); i++) {
    dependency_count[i] = memgraph.nodes[i].inns.size();
  }
  return dependency_count;
}

// check if all the nodes finished executing
bool gpu_execute_state_t::is_complete() {
  auto idx = 0;
  for (auto it = num_nodes_remaining.begin(); it != num_nodes_remaining.end();
       it++) {
    if (it->second != 0) {
      return false;
    }
    idx++;
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

// calling cuda malloc to allocate memory for a given size
float *gpu_allocate_memory(size_t size) {
  void *ret;
  if (cudaMalloc(&ret, size) != cudaSuccess) {
    // print an error message and the error code
    std::cout << "Error code: " << cudaGetLastError() << std::endl;
    throw std::runtime_error("cuda_malloc");
  }
  return (float *)ret;
}

// helper function to get the input memory pointers from a vector of mem_t
// input memory pointers are mem[1: n]
std::vector<float const *> get_input_mem_ptrs(std::vector<mem_t> mem,
                                              float *memory_base_ptr) {
  std::vector<float const *> ret;
  for (int i = 1; i < mem.size(); i++) {
    ret.push_back(offset_increment(memory_base_ptr, mem[i].offset));
  }
  return ret;
}

// get a callback data struct that keeps track of the current node that finished
// execution Has all the data structures required to update things
struct callback_data_t {
  std::mutex *m_ptr;
  std::condition_variable *cv_ptr;
  gpu_execute_state_t *my_state;
  int node_idx;

  void operator()() {
    std::mutex &m = *m_ptr;
    auto &cv = *cv_ptr;
    {
      std::unique_lock lk(m);
      // update the queues since this node is finished
      my_state->finished_queue.push(node_idx);
    }
    cv.notify_all();
  }
};

gpu_execute_state_t::gpu_execute_state_t(memgraph_t const &input_memgraph, float *mem_ptr)
      : memgraph(input_memgraph), memory_base_ptr(mem_ptr) {

	// create a cutensor handle
	HANDLE_ERROR(cutensorCreate(&handle));

	// get the size of the memory needed from the memgraph
	// TODO: hardcoded the 0 since we only have 1 gpu for now
	auto mem_size = memgraph.mem_sizes()[0];

	dependency_count = get_dependencies();
	// in the beginning num_nodes_remaining is the number of nodes in the
	// memgraph
	num_nodes_remaining[0] = memgraph.nodes.size();

	// check all elements from the memgraph and add the nodes with no
	// dependencies to the apply_queue
	for (int i = 0; i < memgraph.nodes.size(); i++) {
		if (memgraph.nodes[i].inns.size() == 0) {
			pending_queue.push(i);
		}
		// check if the node is a contraction
		if (memgraph.nodes[i].op.is_einsummable()) {
			// get the einsummable object
			auto my_einsummable = memgraph.nodes[i].op.get_einsummable();
			// check is this a contraction
			if (my_einsummable.is_contraction()) {
				// merge the adjacent dims
				einsummable_t my_einsum_merged = my_einsummable.merge_adjacent_dims();
				// check if the contraction is already in the map
				if (einsum_to_contraction.find(my_einsum_merged) ==
						einsum_to_contraction.end()) {
					// create a cutensor descriptor
					cutensorContractionDescriptor_t desc;
					// when building the contraction we already merge
					// the adjacent dims so we don't need to do it here
					build_contraction(&desc, handle, my_einsum_merged);
					// add the contraction to the map
					einsum_to_contraction[my_einsum_merged] = desc;
				}
			}
		}
	}

	int num_streams = 500;
	// create a pool of streams
	for (int i = 0; i < num_streams; i++) {
		stream_pool.push(cuda_create_stream());
	}
	// std::cout << "mem_size: " << mem_size << std::endl;
	// allocate memory for the gpu
	memory_base_ptr = mem_ptr;
	// std::cout << "Beginning pending_queue size: " << pending_queue.size() <<
	// std::endl;
}

void execute(const memgraph_t &memgraph, float *memory_base_ptr) {
  // create a gpu_execute_state_t
  gpu_execute_state_t gpu_execute_state(memgraph, memory_base_ptr);
  gpu_execute_state.run();
}

// function definition of gpu_execute_state_t.run()
void gpu_execute_state_t::run() {

  while (true) {

    if (is_complete()) {
      std::cout << "All nodes finished execution." << std::endl;
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
			lk.unlock();
      // std::cout << "Executing node: " << node_idx << std::endl;
      // execute the node
      if (node.op.is_inputmem() || node.op.is_inputsto() || node.op.is_del() ||
          node.op.is_partialize() || node.op.is_alloc()) {
        // do nothing but update the memgraph execution since that node is
        // finished
        std::unique_lock lk(m);
        finished_queue.push(node_idx);
        lk.unlock();
      } else if (node.op.is_apply()) {
        // getting stream from the pool instead of creating a new one
        cudaStream_t stream = cuda_create_stream();
        // get the memory offsets
        auto memory_vector = node.op.get_apply().mems;
        // CASE: TOUCH
        if (node.op.is_touch()) {
          // std::cout << "Got a touch node" << std::endl;
          // TODO: doing a lock; see where is the best place to put this lock
          auto touch = node.op.get_touch();
          auto group_id = node.op.get_apply().group;
          // if we have found this group id in the list, we can't execute until
          // the previous one is done
          if (group_id_executing.count(group_id) != 0) {
            // we can't execute this node since some other node is executing
            // with the same group id add this node to the map
            groupID_to_nodeIDX[group_id].push(node_idx);
            // skipping the callback since this node didn't execute
            continue;
          } else {
            // else we are free to run this
            // see this is the first time seeing this group id
            if (all_group_ids.count(group_id) == 0) {
              // set the castable to nullopt
              touch.castable = std::nullopt;
            } else if (group_id < 0) {
              // set the castable to nullopt
              touch.castable = std::nullopt;
            } else {
              if (touch.castable == std::nullopt) {
                throw std::runtime_error(
                    "Error: Castable is not set for a touch node.");
              }
            }
            // add this group id to the executing set
            group_id_executing.insert(group_id);
            all_group_ids.insert(group_id);
            auto touch_kernel = build_touch(touch);
            touch_kernel(
                stream,
                offset_increment(memory_base_ptr, memory_vector[0].offset),
                offset_increment(memory_base_ptr, memory_vector[1].offset));
          }
        } else {
          auto my_einsummable = node.op.get_einsummable();
          // CASE: CONTRACTION
          if (my_einsummable.is_contraction()) {
            // do a check of the offsets
            checkContractionOffset(node_idx);
            // merge the adjacent dims
            // std::cout << "Got a contraction node" << std::endl;
            einsummable_t my_einsum_merged =
                my_einsummable.merge_adjacent_dims();
            // print an error if we didn't find my_einsum_merged in the map
            auto einsum_iter = einsum_to_contraction.find(my_einsum_merged);
            if (einsum_iter == einsum_to_contraction.end()) {
              throw std::runtime_error(
                  "Error: contraction descriptor not found in the map of "
                  "contraction plans.");
            }
            auto contraction_descriptor = einsum_iter->second;
            execute_contraction(
                stream, handle, &contraction_descriptor,
                offset_increment(memory_base_ptr, memory_vector[0].offset),
                offset_increment(memory_base_ptr, memory_vector[1].offset),
                offset_increment(memory_base_ptr, memory_vector[2].offset));
          }
          // CASE: OTHER EINSUMMABLE
          else {
            // std::cout << "Got a other einsummable node" << std::endl;
            auto cutensor_kernel = build_einsummable(my_einsummable);
            cutensor_kernel(
                stream, handle,
                offset_increment(memory_base_ptr, memory_vector[0].offset),
                get_input_mem_ptrs(memory_vector, memory_base_ptr));
          }
        }

        // after execution, we attach the stream with a callback function
        // get all the metadata needed for the callback
        callback_data_t *data = new callback_data_t;
        data->m_ptr = &m;
        data->cv_ptr = &cv;
        data->node_idx = node_idx;
        data->my_state = this;
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
      } else {
        // print a message saying that the operation is not supported and this
        // operation's type
        throw std::runtime_error("Error: Operation not supported: Type is "
                                 "among the following - move, evict, load");
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
      add_to_pending_queue(new_nodes);
    }
    lk.unlock();
  }
}

// ------------------------ Event loop with multiple streams trying to update
// the same time ------------------------ // struct callback_data_t {
//   std::mutex* m_ptr;
//   std::condition_variable* cv_ptr;
//   gpu_execute_state_t* my_state;
//   int node_idx;

//   void operator()() {
//     std::mutex& m = *m_ptr;
//     auto& cv = *cv_ptr;
//     {
//       std::unique_lock lk(m);
//       // update the queue since this node is finished
//       auto new_nodes = my_state->node_update(node_idx);
//       my_state->add_to_pending_queue(new_nodes);
//     }
//     cv.notify_all();
//   }
// };

// void execute(const memgraph_t& memgraph, float* memory_base_ptr) {
//     // create a gpu_execute_state_t
//     gpu_execute_state_t gpu_execute_state(memgraph, memory_base_ptr);
//     gpu_execute_state.run();
// }

// // function definition of gpu_execute_state_t.run()
// void gpu_execute_state_t::run() {

//     while (true) {

//         if (is_complete()) {
//             std::cout << "All nodes finished execution." << std::endl;
//             break;
//         }

//         // locking the mutex until the queue has new things to execute
//         {
//             std::unique_lock lk(m);
//             cv.wait(lk, [&]{
//                 return pending_queue.size() > 0;
//             });
//         }

//         // execute things that are in the apply_queue until the queue is
//         empty while (pending_queue.size() != 0) {
//             // print out the pending queue
//             // get the first element in the queue
//             auto node_idx = pending_queue.front();
//             auto node = memgraph.nodes[node_idx];
//             // remove the first element from the queue
//             pending_queue.pop();
//             // execute the node
//             if (node.op.is_inputmem() || node.op.is_inputsto() ||
//             node.op.is_del()
//                 || node.op.is_partialize() || node.op.is_alloc()) {
//                 std::unique_lock lk(m);
//                 // do nothing but update the memgraph execution since that
//                 node is finished auto new_nodes = node_update(node_idx);
//                 add_to_pending_queue(new_nodes);
//                 lk.unlock();
//             }
//             else if (node.op.is_apply()) {
//                 // create a cuda stream since for apply we need to execute
//                 that on a cuda stream always
//                 // TODO: may need to keep a pool of streams
//                 cudaStream_t stream = cuda_create_stream();
//                 // get the memory offsets
//                 auto memory_vector = node.op.get_apply().mems;
//                 // CASE: TOUCH
//                 if (node.op.is_touch()) {
//                     // std::cout << "Got a touch node" << std::endl;
//                     // TODO: doing a lock; see where is the best place to put
//                     this lock std::unique_lock lk(m); auto touch =
//                     node.op.get_touch(); auto group_id =
//                     node.op.get_apply().group;
//                     // if we have found this group id in the list, we can't
//                     execute until the previous one is done if
//                     (group_id_executing.count(group_id) != 0) {
//                         // we can't execute this node since some other node
//                         is executing with the same group id
//                         // add this node to the map
//                         groupID_to_nodeIDX[group_id].push(node_idx);
//                         // skipping the callback since this node didn't
//                         execute lk.unlock(); continue;
//                     }
//                     else{
//                         // else we are free to run this
//                         // see this is the first time seeing this group id
//                         if (all_group_ids.count(group_id) == 0){
//                             // set the castable to nullopt
//                             touch.castable = std::nullopt;
//                         }
//                         else if (group_id < 0){
//                             // set the castable to nullopt
//                             touch.castable = std::nullopt;
//                         }
//                         else{
//                             if (touch.castable == std::nullopt){
//                                 throw std::runtime_error("Error: Castable is
//                                 not set for a touch node.");
//                             }
//                         }
//                         // add this group id to the executing set
//                         group_id_executing.insert(group_id);
//                         all_group_ids.insert(group_id);
//                         lk.unlock();
//                         auto touch_kernel = build_touch(touch);
//                         touch_kernel(stream,
//                         offset_increment(memory_base_ptr,
//                         memory_vector[0].offset),
//                         offset_increment(memory_base_ptr,
//                         memory_vector[1].offset));
//                     }
//                 }
//                 else {
//                     auto my_einsummable = node.op.get_einsummable();
//                     // CASE: CONTRACTION
//                     if (my_einsummable.is_contraction()) {
//                         // do a check of the offsets
//                         checkContractionOffset(node_idx);
//                         // merge the adjacent dims
//                         // std::cout << "Got a contraction node" <<
//                         std::endl; einsummable_t my_einsum_merged =
//                         my_einsummable.merge_adjacent_dims();
//                         // print an error if we didn't find my_einsum_merged
//                         in the map auto einsum_iter =
//                         einsum_to_contraction.find(my_einsum_merged); if
//                         (einsum_iter == einsum_to_contraction.end()) {
//                             throw std::runtime_error
//                                 ("Error: contraction descriptor not found in
//                                 the map of contraction plans.");
//                         }
//                         auto contraction_descriptor = einsum_iter->second;
//                         execute_contraction(stream, handle,
//                         &contraction_descriptor,
//                             offset_increment(memory_base_ptr,
//                             memory_vector[0].offset),
//                             offset_increment(memory_base_ptr,
//                             memory_vector[1].offset),
//                             offset_increment(memory_base_ptr,
//                             memory_vector[2].offset));
//                     }
//                     // CASE: OTHER EINSUMMABLE
//                     else {
//                         // std::cout << "Got a other einsummable node" <<
//                         std::endl; auto cutensor_kernel =
//                         build_einsummable(my_einsummable);
//                         cutensor_kernel(stream, handle,
//                         offset_increment(memory_base_ptr,
//                         memory_vector[0].offset),
//                             get_input_mem_ptrs(memory_vector,
//                             memory_base_ptr));
//                     }
//                 }

//                 // after execution, we attach the stream with a callback
//                 function
//                 // get all the metadata needed for the callback
//                 callback_data_t* data = new callback_data_t;
//                 data->m_ptr =& m;
//                 data->cv_ptr =& cv;
//                 data->node_idx = node_idx;
//                 data->my_state = this;
//                 // add the callback
//                 cudaStreamAddCallback(
//                     stream,
//                     [](CUstream_st*, cudaError, void* raw_data) {
//                         callback_data_t* data =
//                         static_cast<callback_data_t*>(raw_data);
//                         callback_data_t& f = *data;
//                         f();
//                         delete data;
//                     },
//                     static_cast<void*>(data),
//                     0
//                 );
//             }
//             else{
//                 // print a message saying that the operation is not supported
//                 and this operation's type throw std::runtime_error
//                     ("Error: Operation not supported: Type is among the
//                     following - move, evict, load");
//             }
//         }

//     }
// }

// ---------------- Calling streams from a stream pool version ----------------
// THIS REDUCES THE OVERHEAD OF CREATING STREAMS, BUT SEEMS TO MAKE THE
// EXECUTION SLOWER DUE TO INCREASED LOGIC struct callback_data_t {
//   std::mutex* m_ptr;
//   std::condition_variable* cv_ptr;
//   gpu_execute_state_t* my_state;
//   int node_idx;
//   cudaStream_t stream;

//   void operator()() {
//     std::mutex& m = *m_ptr;
//     auto& cv = *cv_ptr;
//     {
//       std::unique_lock lk(m);
//       // update the queues since this node is finished
//       my_state->finished_queue.push(node_idx);
//       my_state->finished_streams.push(stream);
//     }
//     cv.notify_all();
//   }
// };

// void execute(const memgraph_t& memgraph, float* memory_base_ptr) {
//     // create a gpu_execute_state_t
//     gpu_execute_state_t gpu_execute_state(memgraph, memory_base_ptr);
//     gpu_execute_state.run();
// }

// // function definition of gpu_execute_state_t.run()
// void gpu_execute_state_t::run() {

//     while (true) {

//         if (is_complete()) {
//             std::cout << "All nodes finished execution." << std::endl;
//             break;
//         }

//         // locking the mutex until the queue has new things to execute
//         {
//             std::unique_lock lk(m);
//             cv.wait(lk, [&]{
//                 // wake up if there are possible updates or new nodes to run
//                 return finished_queue.size() > 0 || pending_queue.size() > 0
//                 || finished_streams.size() > 0;
//             });
//         }

//         // execute things that are in the apply_queue until the queue is
//         empty while (pending_queue.size() != 0) {
//             // print out the pending queue
//             // get the first element in the queue
//             auto node_idx = pending_queue.front();
//             auto node = memgraph.nodes[node_idx];
//             // remove the first element from the queue
//             pending_queue.pop();
//             // std::cout << "Executing node: " << node_idx << std::endl;
//             // execute the node
//             if (node.op.is_inputmem() || node.op.is_inputsto() ||
//             node.op.is_del()
//                 || node.op.is_partialize() || node.op.is_alloc()) {
//                 // do nothing but update the memgraph execution since that
//                 node is finished std::unique_lock lk(m);
//                 finished_queue.push(node_idx);
//                 lk.unlock();
//             }
//             else if (node.op.is_apply()) {
//                 // getting stream from the pool instead of creating a new one
//                 if (stream_pool.size() == 0) {
//                     // we can't do anything since we don't have any streams
//                     available
//                     // add this node to the pending queue and wait for a
//                     stream to be available
//                     // std::cout << "No streams available. Waiting for a
//                     stream to be available." << std::endl;
//                     node_idx_waiting_for_stream.push(node_idx);
//                     continue;
//                 }
//                 cudaStream_t stream = stream_pool.front();
//                 // std::cout << "Node " << node_idx << " got a stream from
//                 the pool" << std::endl; stream_pool.pop();
//                 // get the memory offsets
//                 auto memory_vector = node.op.get_apply().mems;
//                 // CASE: TOUCH
//                 if (node.op.is_touch()) {
//                     // std::cout << "Got a touch node" << std::endl;
//                     // TODO: doing a lock; see where is the best place to put
//                     this lock auto touch = node.op.get_touch(); auto group_id
//                     = node.op.get_apply().group;
//                     // if we have found this group id in the list, we can't
//                     execute until the previous one is done if
//                     (group_id_executing.count(group_id) != 0) {
//                         // we can't execute this node since some other node
//                         is executing with the same group id
//                         // add this node to the map
//                         groupID_to_nodeIDX[group_id].push(node_idx);
//                         // skipping the callback since this node didn't
//                         execute continue;
//                     }
//                     else{
//                         // else we are free to run this
//                         // see this is the first time seeing this group id
//                         if (all_group_ids.count(group_id) == 0){
//                             // set the castable to nullopt
//                             touch.castable = std::nullopt;
//                         }
//                         else if (group_id < 0){
//                             // set the castable to nullopt
//                             touch.castable = std::nullopt;
//                         }
//                         else{
//                             if (touch.castable == std::nullopt){
//                                 throw std::runtime_error("Error: Castable is
//                                 not set for a touch node.");
//                             }
//                         }
//                         // add this group id to the executing set
//                         group_id_executing.insert(group_id);
//                         all_group_ids.insert(group_id);
//                         auto touch_kernel = build_touch(touch);
//                         touch_kernel(stream,
//                         offset_increment(memory_base_ptr,
//                         memory_vector[0].offset),
//                         offset_increment(memory_base_ptr,
//                         memory_vector[1].offset));
//                     }
//                 }
//                 else {
//                     auto my_einsummable = node.op.get_einsummable();
//                     // CASE: CONTRACTION
//                     if (my_einsummable.is_contraction()) {
//                         // do a check of the offsets
//                         checkContractionOffset(node_idx);
//                         // merge the adjacent dims
//                         // std::cout << "Got a contraction node" <<
//                         std::endl; einsummable_t my_einsum_merged =
//                         my_einsummable.merge_adjacent_dims();
//                         // print an error if we didn't find my_einsum_merged
//                         in the map auto einsum_iter =
//                         einsum_to_contraction.find(my_einsum_merged); if
//                         (einsum_iter == einsum_to_contraction.end()) {
//                             throw std::runtime_error
//                                 ("Error: contraction descriptor not found in
//                                 the map of contraction plans.");
//                         }
//                         auto contraction_descriptor = einsum_iter->second;
//                         execute_contraction(stream, handle,
//                         &contraction_descriptor,
//                             offset_increment(memory_base_ptr,
//                             memory_vector[0].offset),
//                             offset_increment(memory_base_ptr,
//                             memory_vector[1].offset),
//                             offset_increment(memory_base_ptr,
//                             memory_vector[2].offset));
//                     }
//                     // CASE: OTHER EINSUMMABLE
//                     else {
//                         // std::cout << "Got a other einsummable node" <<
//                         std::endl; auto cutensor_kernel =
//                         build_einsummable(my_einsummable);
//                         cutensor_kernel(stream, handle,
//                         offset_increment(memory_base_ptr,
//                         memory_vector[0].offset),
//                             get_input_mem_ptrs(memory_vector,
//                             memory_base_ptr));
//                     }
//                 }

//                 // after execution, we attach the stream with a callback
//                 function
//                 // get all the metadata needed for the callback
//                 callback_data_t* data = new callback_data_t;
//                 data->m_ptr =& m;
//                 data->cv_ptr =& cv;
//                 data->node_idx = node_idx;
//                 data->my_state = this;
//                 data->stream = stream;
//                 // add the callback
//                 cudaStreamAddCallback(
//                     stream,
//                     [](CUstream_st*, cudaError, void* raw_data) {
//                         callback_data_t* data =
//                         static_cast<callback_data_t*>(raw_data);
//                         callback_data_t& f = *data;
//                         f();
//                         delete data;
//                     },
//                     static_cast<void*>(data),
//                     0
//                 );
//             }
//             else{
//                 // print a message saying that the operation is not supported
//                 and this operation's type throw std::runtime_error
//                     ("Error: Operation not supported: Type is among the
//                     following - move, evict, load");
//             }
//         }

//         int num_finished = 0;

//         while(finished_queue.size() != 0){
//             // get the node index
//             int node_idx = finished_queue.front();
//             // pop the node index
//             finished_queue.pop();
//             // update the queue since this node is finished
//             auto new_nodes = node_update(node_idx);
//             num_finished = new_nodes.size();
//             add_to_pending_queue(new_nodes);
//         }

//         int num_added = finished_streams.size();

//         while (num_added > 0){
//             if (node_idx_waiting_for_stream.size() == 0){
//                 break;
//             }
//             int node_idx = node_idx_waiting_for_stream.front();
//             node_idx_waiting_for_stream.pop();
//             pending_queue.push(node_idx);
//         }

//         // if we assume each node takes 1 stream (which is not true)
//         // then the number of nodes we can additional insert is:
//         // the number of finished streams - the number of nodes we added to
//         the pending queue above while (finished_streams.size() != 0){
//             cudaStream_t stream = finished_streams.front();
//             finished_streams.pop();
//             stream_pool.push(stream);
//         }
//     }
// }