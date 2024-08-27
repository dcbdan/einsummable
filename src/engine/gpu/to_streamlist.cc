#include "to_streamlist.h"

#include "to_cudagraph.h"

struct streamlist_make_t {
  using op_t = streamlist_t::op_t;

  streamlist_make_t(memgraph_t const& mg, int n);
  memgraph_t const& memgraph;

  vector<op_t> ops;

  int num_streams_per_device;

  vector<int> which_stream;

  int num_events;

  map<int, int> mid_to_oid;

  int get_device(int mid) const {
    auto const& node = memgraph.nodes[mid];    
    if(node.op.is_move()) {
      return node.op.get_move().get_src_loc();
    } else {
      return node.op.get_loc();      
    }
  }

  op_t&       get_op(int mid)       {
    return ops[mid_to_oid.at(mid)];
  }
  op_t const& get_op(int mid) const {
    return ops[mid_to_oid.at(mid)];
  }

  static op_t empty_op(int mid, int device, int stream) {
    return op_t {
      .wait = {},
      .mid = mid,
      .device = device,
      .stream = stream,
      .event = std::nullopt,
      .kernel_info = std::nullopt
    };
  }

  static bool same_stream(op_t const& lhs, op_t const& rhs) {
    return lhs.device == rhs.device && lhs.stream == rhs.stream;
  }

  int next_event() {
    return num_events++;
  }

  void wait_on(op_t& cur, op_t& inn) {
    if(!bool(inn.event)) {
      inn.event = next_event();
    }
    cur.wait.push_back(inn.event.value());
  }

  int get_next_stream(int device) {
    int& which = which_stream[device];
    int ret = which;
    which = (which + 1) % num_streams_per_device;
    return ret;
  }

  void insert_op(int mid) {
    int device = get_device(mid);
    int stream = get_next_stream(device);
    ops.push_back(empty_op(mid, device, stream));
    op_t& cur_op = ops.back();
    for(int const& inn_mid: _get_deps(memgraph, mid)) {
      op_t& inn_op = get_op(inn_mid);
      if(!same_stream(cur_op, inn_op)) {
        wait_on(cur_op, inn_op);
      }
    }
    mid_to_oid.insert({mid, ops.size()-1});
  }
};

streamlist_make_t::streamlist_make_t(memgraph_t const& mg, int n)
  : memgraph(mg)
{
  num_streams_per_device = n;
  int num_devices = memgraph.mem_sizes().size();
  which_stream = vector<int>(num_devices, 0);
  num_events = 0;
}

streamlist_t streamlist_t::make(
  memgraph_t const& memgraph, 
  int num_streams_per_device)
{
  streamlist_make_t make(memgraph, num_streams_per_device);
  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    if(_is_dummy(memgraph, mid)) {
      continue;
    }

    make.insert_op(mid);
  }

  return streamlist_t {
    .memgraph = memgraph,
    .ops = make.ops,
    .num_streams_per_device = num_streams_per_device,
    .num_devices = int(memgraph.mem_sizes().size()),
    .num_events = make.num_events
  };
}

void streamlist_t::compile_kernels(
  vector<kernel_manager_t>& kms,
  map<string, scalar_t> const& scalar_vars)
{
  for(op_t& op: ops) {
    auto const& node = memgraph.nodes[op.mid];
    if(node.op.is_einsummable()) {
      einsummable_t e = node.op.get_einsummable()
        .replace_scalar_variables(scalar_vars)
        .merge_adjacent_dims();

      auto maybe = kms[op.device].build(e);
      if(!maybe) {
        throw std::runtime_error("could not compile a kernel...");
      }

      op.kernel_info = kms[op.device].get_built_kernel_info(e);
    }
  }
}

#define SOUT(x)        if(loud) { DOUT(x);     }
#define SLINE          if(loud) { DLINE;       }
#define SLINEOUT(x)    if(loud) { DLINEOUT(x); }

void streamlist_t::execute(
  vector<kernel_manager_t>& kms,
  vector<vector<cudaStream_t>>& stream_pools,
  vector<void*> mems,
  bool loud) const
{
  vector<cudaEvent_t> events;
  {
    auto start = std::chrono::high_resolution_clock::now();
    vector<cudaEvent_t> es(num_events);
    // Note that each event has to be created on the correct
    // device, so loop through all the events set by the ops...
    for(op_t const& op: ops) {
      if(op.event) {
        handle_cuda_error(cudaSetDevice(op.device));
        handle_cuda_error(cudaEventCreate(&es[op.event.value()]));
      }
    }
    events = std::move(es);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    SOUT("event setup time: " << duration.count() << " ms");
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    for(op_t const& op: ops) {
      cudaStream_t& stream = stream_pools[op.device][op.stream];

      // 1. wait for all the events we need to wait on
      for(int const& wait_event: op.wait) {
        handle_cuda_error(cudaStreamWaitEvent(stream, events[wait_event]));
      }

      // 2. execute the op
      auto const& node = memgraph.nodes[op.mid];
      if(node.op.is_constant()) {
        auto const& info = node.op.get_constant();
        void* out = increment_void_ptr(mems[op.device], info.offset);
       
        if(info.fill.is_constant()) {
          kms[op.device].constant_fill(info.fill.get_constant(), stream, out);
        } else if(info.fill.is_lowertri()) {
          kms[op.device].lowerTri_fill(info.fill.get_lowertri(), stream, out);
        } else {
          throw std::runtime_error("should not reach");
        }
      } else if(node.op.is_touch()) {
        auto const& apply = node.op.get_apply();
        auto const& touch = apply.get_touch();

        if(apply.workspace) {
          throw std::runtime_error("should not have a workspace for touch");
        }

        if(apply.group >= 0) {
          if(!touch.castable) {
            throw std::runtime_error("with group, must have castable!");
          }
        } else {
          if(touch.castable) {
            throw std::runtime_error("should not have a castable here");
          }
        }

        void* out_mem = increment_void_ptr(
          mems[op.device],
          apply.mems[0].offset);

        void const* inn_mem = increment_void_ptr(
          mems[op.device],
          apply.mems[1].offset);

        handle_cuda_error(cudaSetDevice(op.device));
        launch_touch_kernel(touch, stream, out_mem, inn_mem);
      } else if(node.op.is_einsummable()) {
        auto const& apply = node.op.get_apply();
        if(apply.group >= 0) {
          throw std::runtime_error("einsummable should not have group");
        }

        void* global_buffer = mems[op.device];
        void* out_mem = increment_void_ptr(
          global_buffer,
          apply.mems[0].offset);

        vector<void const*> inn_mems;
        inn_mems.reserve(apply.mems.size() - 1);
        for(int i = 1; i != apply.mems.size(); ++i) {
          inn_mems.push_back(increment_void_ptr(
            global_buffer,
            apply.mems[i].offset));
        }

        optional<tuple<void*, uint64_t>> workspace;
        if(apply.workspace) {
          mem_t const& w = apply.workspace.value();
          workspace = {
            increment_void_ptr(global_buffer, w.offset),
            w.size
          };
        }
 
        kms[op.device](
          op.kernel_info.value(),
          stream, 
          out_mem, inn_mems, workspace);
      } else if(node.op.is_move()) {
        auto const& move = node.op.get_move();
        auto const& [src_device, src_offset] = move.src;
        auto const& [dst_device, dst_offset] = move.dst;

        void const* src_mem = increment_void_ptr(
          mems[src_device],
          src_offset);

        void* dst_mem = increment_void_ptr(
          mems[dst_device],
          dst_offset);

        handle_cuda_error(cudaMemcpyAsync(
          dst_mem, src_mem, move.size, 
          cudaMemcpyDeviceToDevice, 
          stream));
      } else {
        throw std::runtime_error("missing node case: streamlist_t::execute");
      }

      // 3. record this event as happened
      if(op.event) {
        handle_cuda_error(cudaEventRecordWithFlags(
          events[op.event.value()],
          stream));
      }
    }

    for(int device = 0; device != num_devices; ++device) {
      handle_cuda_error(cudaSetDevice(device));
      handle_cuda_error(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    SOUT("Streamlist finished. Time: " << duration.count() << " ms");
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    for(cudaEvent_t& event: events) {
      handle_cuda_error(cudaEventDestroy(event));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    SOUT("event destroy time: " << duration.count() << " ms");
  }  
}

