#include "execute.h"

#include <thread>
#include <mutex>
#include <condition_variable>

using std::thread;
using std::queue;

struct applys_progress_t {
  queue<int> ready;

  // once num_remaining[idx] is zero,
  // apply node taskgraph.nodes[idx] can be executed
  map<int, int> num_remaining;

  map<int, vector<int>> input_to_applys;

  // Note that tensor_usage_cnts is passed in
  // by reference.
  void insert(
    int apply_id,
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
    std::optional<touch_info_t> notify_tensor_ready(int tensor_id);

    // Tell the unit it did something so it is no longer be busy.
    // In return, the unit either stays busy returning work to do,
    // or returns nothing and is not busy.
    //
    // If the result is:
    //   none:      the unit MAY be done
    //   something: the unit needs this something to be computed
    //              before it can be done
    std::optional<touch_info_t> completed();

    // return any op that is ready, if not busy
    std::optional<touch_info_t> try_to_pop();
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
  std::optional<int> completed(int unit_id);
};

struct cpu_exec_state_t {
  // collect all the meta data and get this mpi rank ready for execution
  cpu_exec_state_t(taskgraph_t const& taskgraph, map<int, buffer_t>& tensors);

  // launch the threads and wait for the threads to finish
  void run(int n_apply, int n_touch, int n_comm);

  // the threads
  void apply_runner(int runner_id);
  void touch_runner(int runner_id);
  void communicate_runner(int runner_id);

  // update state helper methods
  void _completed(
    bool command_finished,
    vector<int> const& used_as_input,
    std::optional<int> created_tensor);
  void completed_send(int move_id);
  void completed_recv(int move_id);
  void completed_touch(int inn, int unit_id);
  void completed_apply(int apply_id);

  // update the state now that this tensor is
  // available
  void notify_tensor_ready(int tensor_id);

  // misc
  bool check_complete() const;

  // grab tensors (and allocate if necc) under mutex
  tuple<
    vector<buffer_t>,
    vector<buffer_t> >
  get_buffers(
    vector<tuple<uint64_t, int>> const& which_allocate,
    vector<int> const& which_get);

  taskgraph_t const& taskgraph;
  map<int, buffer_t>& tensors;

  // The total number of commands left to execute
  int num_remaining;

  // once num_usages_remaining[idx] is zero,
  // tensors[idx] can be deleted if idx is
  // not a save node
  map<int, int> num_usages_remaining;

  // Concurrency management
  std::mutex m_tensors;
  std::mutex m;
  std::condition_variable cv;

  // Work for apply runners
  applys_progress_t applys_progress;

  // Work for touch runners
  touches_progress_t touches_progress;

  // Work for communicate runners
  // TODO
};

void execute(taskgraph_t const& taskgraph, map<int, buffer_t>& tensors)
{
  cpu_exec_state_t state(taskgraph, tensors);

  // TODO: set n_comm runner > 0
  state.run(8, 8, 0);

  if(!state.check_complete()) {
    throw std::runtime_error("execute did not finish all the tasks");
  }
}

void cpu_exec_state_t::run(int n_apply, int n_touch, int n_comm)
{
  vector<thread> runners;
  runners.reserve(n_apply + n_touch + n_comm);
  for(int i = 0; i != n_apply; ++i) {
    runners.emplace_back([this, i](){ return this->apply_runner(i); });
  }
  for(int i = 0; i != n_touch; ++i) {
    runners.emplace_back([this, i](){ return this->touch_runner(i); });
  }
  for(int i = 0; i != n_comm; ++i) {
    runners.emplace_back([this, i](){ return this->communicate_runner(i); });
  }

  for(auto& t: runners) {
    t.join();
  }
}

cpu_exec_state_t::cpu_exec_state_t(taskgraph_t const& tg, map<int, buffer_t>& ts)
  : taskgraph(tg), tensors(ts), num_remaining(0)
{
  int this_loc = 0;

  // 0. set num_remaining
  // 1. Set num_usages_remaining
  // 2. register every apply node at this location with applys_progress
  // 3. register every partialize node at this location with touches_progress

  vector<int> input_ids;

  int num_nodes = taskgraph.nodes.size();
  for(int id = 0; id != num_nodes; ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_move()) {
      throw std::runtime_error("execute.cc state t: moves not implemented");
    } else if(node.op.is_input()) {
      input_ids.push_back(id);
    } else {
      if(!node.op.output_loc() == this_loc) {
        continue;
      }

      if(node.op.is_apply()) {
        num_remaining += 1;

        applys_progress.insert(
          id,
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
      cv.wait(lk, [this](){return num_remaining == 0 || applys_progress.ready.size() > 0;});
      if(num_remaining == 0) {
        return;
      }
      which = applys_progress.ready.front();
      applys_progress.ready.pop();
    }

    // Do the command execution of which
    {
      // TODO: use kernels other than the reference implementation
      auto const& [_0, inns, einsummable] = taskgraph.nodes[which].op.get_apply();

      auto [_1, inputs] = get_buffers({}, inns);

      auto out_buffer = reference_einsummable(einsummable, inputs);
      std::unique_lock lk(m_tensors);
      tensors.insert({which, out_buffer});
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
        return num_remaining == 0 || touches_progress.ready.size() > 0;
      });
      if(num_remaining == 0) {
        return;
      }
      which = touches_progress.ready.front();
      touches_progress.ready.pop();
    }

    // Do the touch of this operation
    auto const& [unit_id, inn_tensor, touch] = which;

    {
      int partialize_id = touches_progress.units[unit_id].partialize_id;
      uint64_t partialize_size = product(
        vector_from_each_member(touch.selection, uint64_t, d_out));
      auto [_ps, _is] = get_buffers(
        { {partialize_size, partialize_id} },
        { inn_tensor });

      buffer_t& out_buffer = _ps[0];
      buffer_t& inn_buffer = _is[0];

      reference_touch(touch, out_buffer, inn_buffer);
    }

    this->completed_touch(inn_tensor, unit_id);
  }
}

void cpu_exec_state_t::communicate_runner(int runner_id)
{
  // TODO
  throw std::runtime_error("communicate runner not implemented");
}

void cpu_exec_state_t::notify_tensor_ready(int tensor_id)
{
  applys_progress.notify_tensor_ready(tensor_id);
  touches_progress.notify_tensor_ready(tensor_id);
  // TODO for communicate
}

void cpu_exec_state_t::_completed(
  bool command_finished,
  vector<int> const& used_as_input,
  std::optional<int> created_tensor)
{
  if(command_finished) {
    num_remaining -= 1;
  }

  {
    std::unique_lock lk(m_tensors);

    for(auto const& inn: used_as_input) {
      int& cnt = num_usages_remaining.at(inn);
      cnt -= 1;
      if(cnt == 0) {
        num_usages_remaining.erase(inn);
        auto const& inn_node = taskgraph.nodes[inn];
        if(!inn_node.is_save) {
          tensors.erase(inn);
        }
      }
    }
  }

  if(created_tensor) {
    int const& id = created_tensor.value();
    notify_tensor_ready(id);
  }
}

void cpu_exec_state_t::completed_send(int move_id)
{
  auto const& node = taskgraph.nodes[move_id];

  {
    std::unique_lock lk(m);

    _completed(
      true,
      {node.op.get_move().inn},
      std::optional<int>()
    );
  }
  cv.notify_all();
}

void cpu_exec_state_t::completed_recv(int move_id)
{
  {
    std::unique_lock lk(m);

    _completed(
      true,
      {},
      std::optional<int>(move_id)
    );
  }
  cv.notify_all();
}

void cpu_exec_state_t::completed_touch(int inn, int unit_id)
{
  {
    std::unique_lock lk(m);

    std::optional<int> maybe_completed_partial_id =
      touches_progress.completed(unit_id);

    if(maybe_completed_partial_id) {
      auto const& partial_id = maybe_completed_partial_id.value();
      // the partial id has been touched the requisite number of times
      // and so this command is done and partial_id is a completed
      // tensor.
      _completed(
        true,
        {inn},
        std::optional<int>(partial_id)
      );
    } else {
      // More touches need to happen
      _completed(
        false,
        {inn},
        std::optional<int>()
      );
    }
  }
  cv.notify_all();
}

void cpu_exec_state_t::completed_apply(int apply_id)
{
  auto const& node = taskgraph.nodes[apply_id];

  {
    std::unique_lock lk(m);

    set<int> inns = node.op.inputs();
    _completed(
      true,
      vector<int>(inns.begin(), inns.end()),
      std::optional<int>(apply_id)
    );
  }

  cv.notify_all();
}

bool cpu_exec_state_t::check_complete() const {
  // TODO: update for communicate
  // TODO: add anything else to verify?
  return num_remaining == 0;
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
  for(auto const& [sz, id]: which_writes) {
    if(tensors.count(id) == 0) {
      tensors.insert({
        id,
        std::make_shared<buffer_holder_t>(sz)
      });
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
std::optional<int> touches_progress_t::completed(int unit_id) {
  unit_t& unit = units[unit_id];
  auto maybe_op = unit.completed();
  if(maybe_op) {
    ready.push(maybe_op.value());
    // This partial is not complete as this unit needs to
    // wait for the given op
    return std::optional<int>();
  } else if(unit.done()) {
    // Each partial has multiple units it needs to complete,
    // so decrement and see if this unit is the last one
    int& cnt = num_remaining.at(unit.partialize_id);
    cnt -= 1;
    if(cnt == 0) {
      num_remaining.erase(unit.partialize_id);
      return std::optional<int>(unit.partialize_id);
    } else {
      return std::optional<int>();
    }
  } else {
    // This partial is not complete as this unit is not
    // done.
    return std::optional<int>();
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

std::optional<touches_progress_t::touch_info_t>
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

std::optional<touches_progress_t::touch_info_t>
touches_progress_t::unit_t::try_to_pop() {
  if(busy || ready.size() == 0) {
    return std::optional<touch_info_t>();
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
    touch.castable = std::optional<castable_t>();
  }
  return touch_info_t {
    .unit  = unit_id,
    .inn   = inn,
    .touch = touch
  };
}

std::optional<touches_progress_t::touch_info_t>
touches_progress_t::unit_t::completed() {
  if(!busy) {
    throw std::runtime_error("should be busy!");
  }
  busy = false;
  return try_to_pop();
}


