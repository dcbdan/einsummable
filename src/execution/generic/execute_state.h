#include "../../base/setup.h"
#include <cstdint>
#include <memory>
#include "../gpu/kernels.h"
#include "../../einsummable/memgraph.h"
#include "ThreadPool.h"

struct resource_desc_t {
  static resource_desc_t make_empty(); // TODO

  void add_cpu_kernel_manager(); // doesn't actually do anything, probably
  void add_cpu_memory();        // doesn't actually do anything, probably
  void add_cpu_thread();
  void add_cpu_workspace(uint64_t sz);

  void add_gpu_kernel_manager();
  void add_gpu_memory();
  void add_gpu_stream();
  void add_gpu_workspace(uint64_t sz);

};

// The resource manager contains several types of resources
// and acquires subsets of them through try_to_acquire_resources
struct resource_manager_t {
  // TODO TODO: this stuff about resources all needs to be a bit clearer

  // A resources_holder_t has access to a subset of the resources in
  // resource the manager
  struct resources_holder_t {
    // TODO: in destructor make sure that resources are released
    //       or have been released

    void release();

    // For all get_<X> methods, may throw error if this resource object has not
    // been permitted

    // will not throw an error since this "resource" is unlimited
    kernel_manager_t& get_cpu_kernel_manager() { return self->cpu_kernel_manager; }

    gpu_kernel_manager_t& get_gpu_kernel_manager() { }; // TODO

    void launch_on_cpu_thread(std::function<void()> f); // TODO

    void launch_on_gpu_stream(int which_gpu, std::function<void()> f); // TODO
  };


  // TODO
  std::shared_ptr<resources_holder_t>
  try_to_acquire_resources(resource_desc_t const& desc); // TODO

  // --------- cpu resources

  kernel_manager_t* cpu_kernel_manager;

  int num_threads_available;
  ThreadPool threads;

  // --------- gpu resources

  int num_streams_available;
  vector<cudaStream_t> streams;

};

struct exec_graph_t {
  // TODO
  // Build an exec graph ready to run the memgraph on a cpu machine. This graph
  // will only execute the portion of the computation on `this_node`.
  static exec_graph_t make_from_memgraph_with_cpu(
    memgraph_t const& memgraph,
    int this_node);

  // TODO
  // Build an exec graph ready to run the memgraph on gpus. this graph
  // will only execute the porition of the computation on
  // `[this_node*num_gpu_per_node, (this_node+1)*num_gpu_per_node)`.
  static exec_graph_t make_from_memgraph_with_gpu(
    memgraph_t const& memgraph,
    int num_gpu_per_node,
    int this_node);  

  // ---------- misc ops -----------------

  // do nothing op
  struct dummy_t {};

  // ---------- cpu ops  -----------------

  struct einsummable_cpu_t {
    einsummable_t e;
    vector<uint64_t> offsets;
    uint64_t workspace_size;
  };

  struct move_cpu_t {
    // TODO
  };

  
  struct touch_cpu_t {
    // TODO
  };

  // ---------- gpu ops  -----------------

  struct einsummable_gpu_t {
    einsummable_t e;
    vector<mem_t> mems;
  };


  struct touch_gpu_t {
    touch_t t;
    vector<mem_t> mems;
    int group_id;
  };

  struct copy_gpu_t {
    memgraph_t::move_t m;
  };

  // ---------- gpu ops that may need more info --------
  struct evict_gpu_t {
    memloc_t src;
    stoloc_t dst;
  };

  struct load_gpu_t {
    stoloc_t src;
    memloc_t dst;
  };

  using op_t = std::variant<
    dummy_t, einsummable_cpu_t,
    einsummable_gpu_t, touch_cpu_t,
    touch_gpu_t, copy_gpu_t,
    move_cpu_t>;

  // ---------- end of op definitions -----------------
  struct node_t {
    vector<int> outs;
    op_t op;

    resource_desc_t get_resource_desc();

    void launch(
      resource_manager_t::resource_holder_t& resource,
      std::function<void()> callback);

    resource_desc_t required_resources() const;
  };

  node_t gpu_node_translate(memgraph_t::node_t const& n);

  vector<node_t> nodes;
};

struct exec_state_t {
  exec_state_t(exec_graph_t const& g, resource_manager_t& r); // TODO

  // execute all nodes in exec graph
  void event_loop();

  // decrement the output nodes add add them
  // to ready_to_run
  void decrement_outs(int id);

  bool try_to_launch(int id);

  std::queue<int> just_completed;

  vector<int> ready_to_run;

  // for every node in exec graph, the number of dependencies left
  vector<int> num_deps_remaining;

  // total number of things left to do
  int num_remaining;

  // for
  //    just_completed
  std::mutex m_notify;
  std::condition_variable cv_notify;

  exec_graph_t const& exec_graph;
  resource_manager_t& resource_manager;
};

