#include "execute.h"
#include "kernels.h"

#include <thread>
#include <mutex>
#include <condition_variable>

#include <mkl.h> // for mkl_set_num_threads

using std::thread;
using std::queue;

#define RLINEOUT(x) // if(mpi.this_rank == 0) { DLINEOUT(x); }

struct applys_progress_t {
  queue<int> ready;

  // once num_remaining[idx] is zero,
  // apply node taskgraph.nodes[idx] can be executed
  map<int, int> num_remaining;

  map<int, vector<int>> input_to_applys;

  // The goal is for
  //   1. relu(matmul(A,B)) to reuse the matmul(A,B) buffer in the relu
  //   2. C=A+B where A is not used again to become C(:A) += B.
  // In general, any tensor that (1) not a save tensor, (2) is only used once
  // by an elementwise op should be "donatable"..
  // If a tensor is donatable, it's buffer may be consumed by the subsequent
  // elementwise op.
  //
  // This assumes elementwise kernels support donation in all arguments.
  set<int> is_donatable;

  // Note that tensor_usage_cnts is passed in
  // by reference.
  void insert(
    int apply_id,
    bool can_donate,
    set<int> const& inns,
    map<int, int>& tensor_usage_cnts);

  void notify_tensor_ready(int tensor_id);
};

struct touches_progress_t {
  struct touch_info_t {
    int unit;
    int inn;
    touch_t touch;
  };

  struct unit_t {
    unit_t(
      int unit_id,
      int partialize_id,
      vector<tuple<int, touch_t>> const& ops);

    int const unit_id;
    int const partialize_id;
    vector<tuple<int, touch_t>> const ops;

    bool init;
    bool busy;
    vector<int> ready;
    vector<int> waiting;

    // if there is nothing left to compute for this unit
    bool done() const;

    // let this know the tensor id is ready
    optional<touch_info_t> notify_tensor_ready(int tensor_id);

    // Tell the unit it did something so it is no longer be busy.
    // In return, the unit either stays busy returning work to do,
    // or returns nothing and is not busy.
    //
    // If the result is:
    //   none:      the unit MAY be done
    //   something: the unit needs this something to be computed
    //              before it can be done
    optional<touch_info_t> completed();

    // return any op that is ready, if not busy
    optional<touch_info_t> try_to_pop();
  };

  // the touches that can be done
  queue<touch_info_t> ready;
  // Every unit may only have up to one op in the
  // touch_ready queue. If it has an op in the queue
  // or being computed, the busy flag should be set
  // to true

  vector<unit_t> units;

  // input id to unit ids
  map<int, set<int>> input_to_units;

  // partialize id to number of units left to complete
  map<int, int> num_remaining;

  // For initliazation: insert a new partialize op
  //
  // Note that tensor_usage_cnts is passed in by reference.
  // Since each partializes touches are being traversed,
  // increment tensor_usage_cnts each time it is (will be)
  // used by a unit
  void insert_partialize(
    int partialize_id,
    vector<vector<tuple<int, touch_t>>> const& touches,
    map<int, int>& tensor_usage_cnts);

  // notify that this tensor is ready to be used
  void notify_tensor_ready(int input_id);

  // This unit just finished an op. Return
  // the corresponding partial if the partial just
  // completed.
  optional<int> completed(int unit_id);
};

struct sends_progress_t {
  // move ids that are pending
  queue<int> ready;

  // inn id to moves id
  map<int, vector<int>> waiting;

  void insert(int move_id, int inn_id);
  void notify_tensor_ready(int inn_id);
};

struct cpu_exec_state_t {
  // collect all the meta data and get this mpi rank ready for execution
  cpu_exec_state_t(
    mpi_t& mpi,
    taskgraph_t const& taskgraph,
    kernel_manager_t const& kernel_manager,
    map<int, buffer_t>& tensors,
    int num_apply_kernel_threads);

  // launch the threads and wait for the threads to finish
  void run(int n_apply, int n_touch, int n_send, int n_recv);

  // the threads
  void apply_runner(int runner_id);
  void touch_runner(int runner_id);
  void send_runner(int runner_id);
  void recv_runner(int runner_id);

  // update state helper methods
  // must start with lk holding a lock on mutex m and
  // finish with mutex m unlocked
  void _completed(
    std::unique_lock<std::mutex>&& lk, // this must hold mutex m!
    bool command_finished,
    vector<int> const& used_as_input,
    optional<int> created_tensor);
  void completed_send(int move_id);
  void completed_recv(int move_id);
  void completed_touch(int inn, int unit_id);
  void completed_apply(int apply_id);

  // update the state now that this tensor is
  // available
  void notify_tensor_ready(int tensor_id);

  // misc
  bool check_complete() const;

  tuple<
    optional<tuple<void*, uint64_t>>,
    optional<int>>
  get_workspace(einsummable_t const&);

  void release_workspace(int which);
  void release_workspace(optional<int> w) {
    if(w) { return release_workspace(w.value()); }
  }

  // grab tensors (and allocate if necc) under mutex
  tuple<
    vector<buffer_t>,
    vector<buffer_t> >
  get_buffers(
    vector<tuple<uint64_t, int>> const& which_allocate,
    vector<int> const& which_get);

  mpi_t& mpi;
  taskgraph_t const& taskgraph;
  kernel_manager_t const& kernel_manager;
  map<int, buffer_t>& tensors;
  int const num_apply_kernel_threads;

  // The total number of commands left to execute
  int num_remaining;

  // once num_usages_remaining[idx] is zero,
  // tensors[idx] can be deleted if idx is
  // not a save node
  map<int, int> num_usages_remaining;

  // Concurrency management
  std::mutex m_workspace;
  std::mutex m_tensors;
  std::mutex m;
  std::condition_variable cv;

  // Work for apply runners
  applys_progress_t applys_progress;

  // Work for touch runners
  touches_progress_t touches_progress;

  // Work for send and recv runners
  sends_progress_t sends_progress;
  int num_recv_post_remaining;

  // Workspace management
  vector<tuple<bool,buffer_t>> workspace;
};

void execute(
  taskgraph_t const& taskgraph,
  settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t& mpi,
  map<int, buffer_t>& tensors)
{
  cpu_exec_state_t state(
    mpi, taskgraph, kernel_manager, tensors,
    settings.num_apply_kernel_threads);

  state.run(
    settings.num_apply_runner,
    settings.num_touch_runner,
    mpi.world_size > 1 ? settings.num_send_runner : 0,
    mpi.world_size > 1 ? settings.num_recv_runner : 0);

  if(!state.check_complete()) {
    throw std::runtime_error("execute did not finish all the tasks");
  }
}

void cpu_exec_state_t::run(int n_apply, int n_touch, int n_send, int n_recv)
{
  vector<thread> runners;
  runners.reserve(n_apply + n_touch + n_send + n_recv);
  for(int i = 0; i != n_apply; ++i) {
    runners.emplace_back([this, i](){ return this->apply_runner(i); });
  }
  for(int i = 0; i != n_touch; ++i) {
    runners.emplace_back([this, i](){ return this->touch_runner(i); });
  }
  for(int i = 0; i != n_send; ++i) {
    runners.emplace_back([this, i](){ return this->send_runner(i); });
  }
  for(int i = 0; i != n_recv; ++i) {
    runners.emplace_back([this, i](){ return this->recv_runner(i); });
  }

  for(auto& t: runners) {
    t.join();
  }
}

cpu_exec_state_t::cpu_exec_state_t(
  mpi_t& mpi,
  taskgraph_t const& tg,
  kernel_manager_t const& km,
  map<int, buffer_t>& ts,
  int n_ts)
  : mpi(mpi),
    taskgraph(tg),
    kernel_manager(km),
    tensors(ts),
    num_remaining(0),
    num_recv_post_remaining(0),
    num_apply_kernel_threads(n_ts)
{
  // .. tell mkl how many threads to use
  // 0. set num_remaining
  // 1. Set num_usages_remaining
  // 2. register every apply node at this location with applys_progress
  // 3. register every partialize node at this location with touches_progress
  // 4. register every send from here
  // 5. register every recv to   here

  mkl_set_num_threads(num_apply_kernel_threads);

  vector<int> input_ids;

  int num_nodes = taskgraph.nodes.size();
  for(int id = 0; id != num_nodes; ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_move()) {
      auto const& [src,dst,inn,_] = node.op.get_move();

      if(src == dst) {
        throw std::runtime_error("Moves to self are not allowed");
      }

      if(src == mpi.this_rank) {
        // This is a send
        num_remaining += 1;
        num_usages_remaining[inn] += 1;
        sends_progress.insert(id, inn);
      } else if(dst == mpi.this_rank) {
        // This is a recv
        num_remaining += 1;
        num_recv_post_remaining += 1;
      }
    } else {
      // Only do stuff on this location!
      if(node.op.out_loc() != mpi.this_rank) {
        continue;
      }

      if(node.op.is_input()) {
        input_ids.push_back(id);
      } else if(node.op.is_apply()) {
        num_remaining += 1;

        // Can this be marked as donated?
        // 1. not a save node
        // 2. used only once
        // 3. usage node is straight-elementwise
        bool can_donate = false;
        if(!node.is_save && node.outs.size() == 1) {
          int const& node_out_id = *node.outs.begin();
          auto const& node_out = taskgraph.nodes[node_out_id];
          if(node_out.op.is_apply()) {
            auto const& einsummable = node_out.op.get_apply().einsummable;
            can_donate = einsummable.is_straight_elementwise();
          }
        }

        applys_progress.insert(
          id,
          can_donate,
          node.op.inputs(),
          num_usages_remaining
        );
      } else if(node.op.is_partialize()) {
        num_remaining += 1;

        // pass in num_usages_remaining by reference
        touches_progress.insert_partialize(
          id,
          node.op.get_touches(),
          num_usages_remaining
        );
      } else {
        throw std::runtime_error("should not reach");
      }
    }
  }

  // now that everything is setup, state
  // what input tensors are ready initially
  for(auto const& input_id: input_ids) {
    notify_tensor_ready(input_id);
  }
}

void cpu_exec_state_t::apply_runner(int runner_id)
{
  int which;
  while(true)
  {
    // Get a command that needs to be executed, or return
    {
      std::unique_lock lk(m);
      cv.wait(lk, [this](){
        return applys_progress.num_remaining.size() == 0 ||
               applys_progress.ready.size() > 0;
      });
      if(applys_progress.ready.size() > 0) {
        which = applys_progress.ready.front();
        applys_progress.ready.pop();
      } else {
        return;
      }
    }

    // Do the command execution of which
    {
      auto const& node = taskgraph.nodes[which];
      auto const& [_0, inns, einsummable] = node.op.get_apply();

      auto [workspace, which_workspace] = get_workspace(einsummable);

      auto [_1, inputs] = get_buffers({}, inns);

      buffer_t out_buffer;

      // Can we donate one of the input buffers to
      // this computation?
      for(int i = 0; i != inns.size(); ++i) {
        int const& inn = inns[i];
        buffer_t& input = inputs[i];
        if(applys_progress.is_donatable.count(inn)) {
          out_buffer = input;
          break;
        }
      }

      // If not, allocate.
      if(!out_buffer) {
        out_buffer = make_buffer(node.op.out_size());
      }

      vector<void const*> raw_inputs;
      raw_inputs.reserve(inputs.size());
      for(auto const& buffer: inputs) {
        raw_inputs.push_back(buffer->data);
      }

      kernel_manager(einsummable, out_buffer->data, raw_inputs, workspace);
      release_workspace(which_workspace);

      // Note: Even if out_buffer was donated, this is fine. When
      //       the donated input gets removed from tensors, the
      //       buffer won't get deleted since its a shared pointer.
      std::unique_lock lk(m_tensors);
      tensors.insert_or_assign(which, out_buffer);
    }

    this->completed_apply(which);
  }
}

void cpu_exec_state_t::touch_runner(int runner_id)
{
  using touch_info_t = touches_progress_t::touch_info_t;
  touch_info_t which;
  while(true) {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [this](){
        return touches_progress.num_remaining.size() == 0 ||
               touches_progress.ready.size() > 0;
      });
      if(touches_progress.ready.size() > 0) {
        which = touches_progress.ready.front();
        touches_progress.ready.pop();
      } else {
        return;
      }
    }

    // Do the touch of this operation
    auto const& [unit_id, inn_tensor, touch] = which;

    {
      int partialize_id = touches_progress.units[unit_id].partialize_id;
      uint64_t partialize_size = taskgraph.nodes[partialize_id].op.out_size();
      auto [_ps, _is] = get_buffers(
        { {partialize_size, partialize_id} },
        { inn_tensor });

      buffer_t& out_buffer = _ps[0];
      buffer_t& inn_buffer = _is[0];

      kernel_manager(touch, out_buffer->data, inn_buffer->data);
    }

    this->completed_touch(inn_tensor, unit_id);
  }
}

void cpu_exec_state_t::send_runner(int runner_id)
{
  int send_id;

  while(true)
  {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [this](){
        return sends_progress.waiting.size() == 0 ||
               sends_progress.ready.size() > 0;
      });
      if(sends_progress.ready.size() > 0) {
        send_id = sends_progress.ready.front();
        sends_progress.ready.pop();
      } else {
        return;
      }
    }

    auto const& node = taskgraph.nodes[send_id];
    auto const& [_0, dst, inn_id, _1] = node.op.get_move();

    mpi.send_int(send_id, dst, mpi.max_tag);

    auto [_, read_buffers] = get_buffers({}, {inn_id});
    auto& buffer = read_buffers[0];

    mpi.send(buffer, dst, send_id);

    completed_send(send_id);
  }
}

void cpu_exec_state_t::recv_runner(int runner_id)
{
  while(true)
  {
    {
      std::unique_lock lk(m);

      if(num_recv_post_remaining == 0) {
        return;
      }
      num_recv_post_remaining -= 1;
      // (don't do this after here because if there are two
      //  recv runners for one post remaining, then mpi recv could be
      //  called twice waiting for the same data and would would hang)
    }

    int recv_id = mpi.recv_int_from_anywhere(mpi.max_tag);

    auto const& node = taskgraph.nodes[recv_id];
    auto const& [src, _0, _1, size] = node.op.get_move();

    auto [should_be_new_buffers, _] = get_buffers({ {size, recv_id} }, {});
    auto& recv_buffer = should_be_new_buffers[0];

    mpi.recv(recv_buffer, src, recv_id);

    completed_recv(recv_id);
  }
}

void cpu_exec_state_t::notify_tensor_ready(int tensor_id)
{
  applys_progress.notify_tensor_ready(tensor_id);
  touches_progress.notify_tensor_ready(tensor_id);
  sends_progress.notify_tensor_ready(tensor_id);
}

void cpu_exec_state_t::_completed(
  std::unique_lock<std::mutex>&& lk, // this must hold mutex m!
  bool command_finished,
  vector<int> const& used_as_input,
  optional<int> created_tensor)
{
  if(command_finished) {
    num_remaining -= 1;
  }

  vector<int> will_erase;
  for(auto const& inn: used_as_input) {
    int& cnt = num_usages_remaining.at(inn);
    cnt -= 1;
    if(cnt == 0) {
      num_usages_remaining.erase(inn);
      auto const& inn_node = taskgraph.nodes[inn];
      if(!inn_node.is_save) {
        will_erase.push_back(inn);
      }
    }
  }

  if(created_tensor) {
    int const& id = created_tensor.value();
    notify_tensor_ready(id);
  }

  // we don't need this lock to delete from tenors,
  // so free it up
  lk.unlock();

  if(will_erase.size() > 0) {
    // but we do need a lock over tensors
    std::unique_lock lk_tensors(m_tensors);
    for(auto const& inn: will_erase) {
      tensors.erase(inn);
    }
  }
}

void cpu_exec_state_t::completed_send(int move_id)
{
  auto const& node = taskgraph.nodes[move_id];

  _completed(
    std::unique_lock(m),
    true,
    {node.op.get_move().inn},
    optional<int>()
  );

  cv.notify_all();
}

void cpu_exec_state_t::completed_recv(int move_id)
{
  _completed(
    std::unique_lock(m),
    true,
    {},
    optional<int>(move_id)
  );

  cv.notify_all();
}

void cpu_exec_state_t::completed_touch(int inn, int unit_id)
{
  std::unique_lock lk(m);

  optional<int> maybe_completed_partial_id =
    touches_progress.completed(unit_id);

  if(maybe_completed_partial_id) {
    auto const& partial_id = maybe_completed_partial_id.value();
    // the partial id has been touched the requisite number of times
    // and so this command is done and partial_id is a completed
    // tensor.
    _completed(
      std::move(lk),
      true,
      {inn},
      optional<int>(partial_id)
    );
  } else {
    // More touches need to happen
    _completed(
      std::move(lk),
      false,
      {inn},
      optional<int>()
    );
  }
  // _completed has release the mutex

  cv.notify_all();
}

void cpu_exec_state_t::completed_apply(int apply_id)
{
  auto const& node = taskgraph.nodes[apply_id];
  set<int> inns = node.op.inputs();


  _completed(
    std::unique_lock(m),
    true,
    vector<int>(inns.begin(), inns.end()),
    optional<int>(apply_id)
  );


  cv.notify_all();
}

bool cpu_exec_state_t::check_complete() const {
  return num_remaining == 0;
}

tuple<
  optional<tuple<void*, uint64_t>>,
  optional<int>>
cpu_exec_state_t::get_workspace(einsummable_t const& e)
{
  uint64_t size = kernel_manager.workspace_size(e);
  if(size == 0) {
    return {std::nullopt, std::nullopt};
  }
  std::unique_lock lk(m_workspace);
  int ret = -1;
  int smallest;
  for(int i = 0; i != workspace.size(); ++i) {
    auto const& [is_available, buffer] = workspace[i];
    uint64_t const& sz = buffer->size;
    if(is_available && size <= sz) {
      if(ret == -1) {
        ret = i;
        smallest = sz;
      } else {
        if(sz < smallest) {
          ret = i;
          smallest = sz;
        }
      }
    }
  }
  if(ret == -1) {
    workspace.emplace_back(false, make_buffer(size));
    ret = workspace.size()-1;
  } else {
    auto& [is_available, _] = workspace[ret];
    is_available = false;
  }

  auto b = std::get<1>(workspace[ret]);
  using t1 = optional<tuple<void*, uint64_t>>;
  using t2 = optional<int>;
  return tuple<t1,t2>{
    t1({b->data, b->size}),
    t2(ret)
  };
}

void cpu_exec_state_t::release_workspace(int which) {
  std::unique_lock lk(m_workspace);
  auto& [is_available, _] = workspace[which];
  is_available = true;
}

tuple<
  vector<buffer_t>,
  vector<buffer_t> >
cpu_exec_state_t::get_buffers(
  vector<tuple<uint64_t, int>> const& which_writes,
  vector<int>                  const& which_reads)
{
  std::unique_lock lk(m_tensors);

  vector<buffer_t> writes;
  writes.reserve(which_writes.size());
  for(auto const& [size, id]: which_writes) {
    if(tensors.count(id) == 0) {
      tensors.insert_or_assign(
        id,
        make_buffer(size)
      );
    }
    writes.push_back(tensors.at(id));
  }

  vector<buffer_t> reads;
  reads.reserve(which_reads.size());
  for(auto const& id: which_reads) {
    reads.push_back(tensors.at(id));
  }

  return {writes, reads};
}

void applys_progress_t::insert(
  int apply_id,
  bool can_donate,
  set<int> const& inns,
  map<int, int>& tensor_usage_cnts)
{
  if(num_remaining.count(apply_id) > 0) {
    throw std::runtime_error("how come applys progress num rem already has this?");
  }

  num_remaining.insert({apply_id, inns.size()});

  for(auto const& inn: inns) {
    input_to_applys[inn].push_back(apply_id);
    tensor_usage_cnts[inn] += 1;
  }

  if(can_donate) {
    is_donatable.insert(apply_id);
  }
}

void applys_progress_t::notify_tensor_ready(int input_id)
{
  if(input_to_applys.count(input_id) > 0) {
    for(auto const& apply_id: input_to_applys.at(input_id)) {
      int& cnt = num_remaining.at(apply_id);
      cnt -= 1;
      if(cnt == 0) {
        ready.push(apply_id);
        num_remaining.erase(apply_id);
      }
    }
  }
}

void touches_progress_t::insert_partialize(
  int partialize_id,
  vector<vector<tuple<int, touch_t>>> const& touches,
  map<int, int>& tensor_usage_cnts)
{
  if(num_remaining.count(partialize_id) > 0) {
    throw std::runtime_error("how come touches progress num rem already has this?");
  }

  num_remaining.insert({partialize_id, touches.size()});

  for(vector<tuple<int, touch_t>> const& ts: touches) {
    int unit_id = units.size();
    units.emplace_back(unit_id, partialize_id, ts);

    for(auto const& [inn, _]: ts) {
      input_to_units[inn].insert(unit_id);
      tensor_usage_cnts[inn] += 1;
    }
  }
}

void touches_progress_t::notify_tensor_ready(int input_id) {
  if(input_to_units.count(input_id) > 0) {
    for(auto const& unit_id: input_to_units.at(input_id)) {
      auto maybe_op = units[unit_id].notify_tensor_ready(input_id);
      if(maybe_op) {
        ready.push(maybe_op.value());
      }
    }
    input_to_units.erase(input_id);
  }
}

// Return the partial id that just completed, if applicable
optional<int> touches_progress_t::completed(int unit_id) {
  unit_t& unit = units[unit_id];
  auto maybe_op = unit.completed();
  if(maybe_op) {
    ready.push(maybe_op.value());
    // This partial is not complete as this unit needs to
    // wait for the given op
    return optional<int>();
  } else if(unit.done()) {
    // Each partial has multiple units it needs to complete,
    // so decrement and see if this unit is the last one
    int& cnt = num_remaining.at(unit.partialize_id);
    cnt -= 1;
    if(cnt == 0) {
      num_remaining.erase(unit.partialize_id);
      return optional<int>(unit.partialize_id);
    } else {
      return optional<int>();
    }
  } else {
    // This partial is not complete as this unit is not
    // done.
    return optional<int>();
  }

}

touches_progress_t::unit_t::unit_t(
  int unit_id,
  int partialize_id,
  vector<tuple<int, touch_t>> const& ops)
  : unit_id(unit_id),
    partialize_id(partialize_id),
    ops(ops),
    init(true),
    busy(false)
{
  // in waiting, store all 0,1, ... ops.size()-1,
  // the op idxs still to do
  waiting = vector<int>(ops.size());
  std::iota(waiting.begin(), waiting.end(), 0);
}

bool touches_progress_t::unit_t::done() const {
  return ready.size() == 0 && waiting.size() == 0;
}

optional<touches_progress_t::touch_info_t>
touches_progress_t::unit_t::notify_tensor_ready(int tensor_id) {
  for(int i = 0; i != ops.size(); ++i) {
    auto const& inn_id = std::get<0>(ops[i]);
    if(inn_id == tensor_id) {
      vector_erase_value(waiting, i);
      ready.push_back(i);
    }
  }
  return try_to_pop();
}

optional<touches_progress_t::touch_info_t>
touches_progress_t::unit_t::try_to_pop() {
  if(busy || ready.size() == 0) {
    return optional<touch_info_t>();
  }

  busy = true;

  int which_op = ready.back();
  ready.pop_back();

  auto [inn, touch] = ops[which_op];

  if(init) {
    // On the first touch of this unit,
    // we want to initialize the output instead of
    // incrementing. So set the castable
    // to none to specify this op.
    init = false;
    touch.castable = optional<castable_t>();
  }
  return touch_info_t {
    .unit  = unit_id,
    .inn   = inn,
    .touch = touch
  };
}

optional<touches_progress_t::touch_info_t>
touches_progress_t::unit_t::completed() {
  if(!busy) {
    throw std::runtime_error("should be busy!");
  }
  busy = false;
  return try_to_pop();
}

void sends_progress_t::insert(int move_id, int inn_id) {
  waiting[inn_id].push_back(move_id);
}

void sends_progress_t::notify_tensor_ready(int inn_id) {
  if(waiting.count(inn_id) > 0) {
    // push the move id(s) onto ready
    for(auto const& move_id: waiting.at(inn_id)) {
      ready.push(move_id);
    }

    waiting.erase(inn_id);
  }
}

kernel_manager_t make_kernel_manager(taskgraph_t const& taskgraph)
{
  kernel_manager_t ret;

  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& e = node.op.get_apply().einsummable;
      if(!ret.build(e)) {
        throw std::runtime_error(
          "could not build a kernel for " + write_with_ss(e));
      }
    }
  }

  return ret;
}

