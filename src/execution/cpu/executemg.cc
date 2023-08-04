#include "executemg.h"
#include "workspace.h"

#include <thread>
#include <mutex>
#include <condition_variable>

using std::thread;
using std::queue;

#define RLINEOUT(x) // if(mpi->this_rank == 0) { DLINEOUT(x); }

void update_kernel_manager(
  kernel_manager_t& ret,
  memgraph_t const& memgraph)
{
  for(auto const& node: memgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      if(apply.is_einsummable()) {
        auto const& e = apply.get_einsummable();
        if(!ret.build(e)) {
          throw std::runtime_error(
            "could not build a kernel for " + write_with_ss(e));
        }
      }
    }
  }
}

struct cpu_mg_exec_state_t {
  cpu_mg_exec_state_t(
    mpi_t* mpi,
    memgraph_t const& memgraph,
    kernel_manager_t const& kernel_manager,
    buffer_t& buffer,
    storage_t* storage);

  void apply_runner(int runner_id);
  void storage_runner(int runner_id);
  void send_runner(int runner_id);
  void recv_runner(int runner_id);

  // these should be launched just once
  void send_can_recv_runner();
  void recv_can_recv_runner();

  void completed(int node_id, int group_id = -1);

  void run(int n_apply, int n_sto, int n_send, int n_recv);

  void* get_raw(mem_t const& m) {
    return reinterpret_cast<void*>(buffer->data + m.offset);
  }
  void* get_raw(memloc_t const& m) {
    return reinterpret_cast<void*>(buffer->data + m.offset);
  }
  buffer_t get_buffer_reference(mem_t const& m) {
    return make_buffer_reference(buffer->data + m.offset, m.size);
  }

  mpi_t* mpi;
  memgraph_t const& memgraph;
  kernel_manager_t const& kernel_manager;
  buffer_t& buffer;
  storage_t* storage;

  // this holds the kernel_manager and queries it
  workspace_manager_t workspace_manager;

  int this_rank;

  std::mutex m;
  std::condition_variable cv;

  // this is treated as a first in first out queue,
  // but different group ids have to be worried about
  // so it isn't strictly fifo.
  vector<int> apply_ready;

  // these can be sent right away
  queue<int> send_ready;

  // the queue to tell the opposing send that
  // the recv here can proceed
  queue<int> can_recv_ready;

  queue<int> sto_ready;

  int num_remaining;

  map<int, int> num_deps_remaining;

  int num_recv_remaining;
  int num_recv_can_recv_remaining;

  set<int> busy_groups;

  // group id -> is it first
  std::unordered_map<int, bool> is_firsts;
};

void _execute_memgraph(
  memgraph_t const& memgraph,
  execute_memgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the memgraph must be single-node
  buffer_t memory,
  storage_t* storage)
{
  // TODO: For shared storage, the memgraph execution engine will need
  // inform the load at dst that the evict at src has happened on the
  // same shared storage location.
  //
  // For now, enforce no shared storage by having storage_loc[i] == i for all i.
  auto const& storage_loc = memgraph.storage_locs;
  for(int i = 0; i != storage_loc.size(); ++i) {
    if(storage_loc[i] != i) {
      throw std::runtime_error("storage locs must be 0,1,...");
    }
  }

  cpu_mg_exec_state_t state(mpi, memgraph, kernel_manager, memory, storage);

  int world_size = bool(mpi) ? mpi->world_size : 1;

  state.run(
    settings.num_apply_runner,
    bool(storage) ? settings.num_storage_runner : 0,
    world_size > 1 ? settings.num_send_runner : 0,
    world_size > 1 ? settings.num_recv_runner : 0);
}

void execute_memgraph(
  memgraph_t const& memgraph,
  execute_memgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi,
  buffer_t memory)
{
  _execute_memgraph(memgraph, settings, kernel_manager, mpi, memory, nullptr);
}

void execute_memgraph(
  memgraph_t const& memgraph,
  execute_memgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the memgraph must be single-node
  buffer_t memory,
  storage_t& storage)
{
  _execute_memgraph(memgraph, settings, kernel_manager, mpi, memory, &storage);
}

void cpu_mg_exec_state_t::run(int n_apply, int n_sto, int n_send, int n_recv)
{
  vector<thread> runners;
  runners.reserve(n_apply + n_sto + n_send + n_recv + 2);
  for(int i = 0; i != n_apply; ++i) {
    runners.emplace_back([this, i](){ return this->apply_runner(i); });
  }
  for(int i = 0; i != n_sto; ++i) {
    runners.emplace_back([this, i](){ return this->storage_runner(i); });
  }
  for(int i = 0; i != n_send; ++i) {
    runners.emplace_back([this, i](){ return this->send_runner(i); });
  }
  for(int i = 0; i != n_recv; ++i) {
    runners.emplace_back([this, i](){ return this->recv_runner(i); });
  }

  if(n_send > 0 || n_recv > 0) {
    runners.emplace_back([this](){ return this->recv_can_recv_runner(); });
    runners.emplace_back([this](){ return this->send_can_recv_runner(); });
  }

  for(auto& t: runners) {
    t.join();
  }
}

cpu_mg_exec_state_t::cpu_mg_exec_state_t(
  mpi_t* mpi,
  memgraph_t const& mg,
  kernel_manager_t const& km,
  buffer_t& b,
  storage_t* ss)
  : mpi(mpi),
    memgraph(mg),
    kernel_manager(km),
    buffer(b),
    workspace_manager(km),
    storage(ss)
{
  // set num_remaining, num_deps_remaining and the move notification setup variables
  vector<int> readys;

  int num_nodes = memgraph.nodes.size();

  this_rank = bool(mpi) ? mpi->this_rank : 0;
  num_remaining = 0;
  num_recv_remaining = 0;

  num_recv_can_recv_remaining = 0;
  int& num_sends = num_recv_can_recv_remaining;

  for(int id = 0; id != num_nodes; ++id) {
    auto const& node = memgraph.nodes[id];
    if(!node.op.is_local_to(this_rank)) {
      continue;
    }

    num_remaining++;

    int num_deps;
    if(node.op.is_move()) {
      auto const& move = node.op.get_move();

      num_deps = 0;
      for(auto const& inn: node.inns) {
        auto const& inn_node = memgraph.nodes[inn];
        if(inn_node.op.is_local_to(this_rank)) {
          num_deps++;
        }
      }

      bool is_send = (this_rank == move.get_src_loc());
      if(is_send) {
        num_deps++;
        num_sends++;
      } else {
        num_recv_remaining++;
      }
    } else {
      // It must be the case that all the inns are local
      num_deps = node.inns.size();

      // Note: if one of the inns is a recving move node, then the
      //       dependency is a bit weird. But it's being allowed
    }

    // num_deps == the number of local dependencies unless
    // the node is a send node, in which case num_deps == num_local_deps + 1
    num_deps_remaining.insert({id, num_deps});

    if(num_deps == 0) {
      if(node.op.is_move()) {
        can_recv_ready.push(id);
      } else {
        readys.push_back(id);
      }
    }

    // fill in is_firsts
    if(node.op.is_apply()) {
      int const& group = node.op.get_apply().group;
      if(group >= 0) {
        is_firsts.insert({group, true});
      }
    }
  }

  // now that everything is setup, get it started
  // with the things that are immediately ready
  for(auto const& ready_id: readys) {
    completed(ready_id);
  }
}

void cpu_mg_exec_state_t::completed(int _node_id, int group_id)
{
  {
    std::unique_lock lk(m);

    if(group_id >= 0) {
      bool did_remove = busy_groups.erase(group_id);
      if(!did_remove) {
        throw std::runtime_error("this group id was not busy!");
      }
    }

    vector<int> node_ids({_node_id});
    while(node_ids.size() > 0) {
      int node_id = node_ids.back();
      node_ids.pop_back();

      // this node has officially been completed,
      // so decrement num_remaining and the necc
      // deps remaining

      num_remaining--;

      auto const& outs = memgraph.nodes[node_id].outs;
      for(auto const& out: outs) {
        if(!memgraph.nodes[out].op.is_local_to(this_rank)) {
          continue;
        }

        auto iter = num_deps_remaining.find(out);
        if(iter == num_deps_remaining.end()) {
          throw std::runtime_error("no dep rem!");
        }
        int& cnt = iter->second;
        cnt--;
        if(cnt == 0) {
          auto const& node = memgraph.nodes[out];

          num_deps_remaining.erase(iter);

          if(node.op.is_apply()) {
            apply_ready.push_back(out);
          } else if(node.op.is_move()) {
            auto const& move = node.op.get_move();
            bool is_send = (this_rank == move.get_src_loc());
            if(is_send) {
              send_ready.push(out);
            } else {
              can_recv_ready.push(out);
            }
          } else if(node.op.is_evict() || node.op.is_load()) {
            sto_ready.push(out);
          } else if(node.op.is_inputmem() || node.op.is_inputsto()
                 || node.op.is_partialize() || node.op.is_alloc()
                 || node.op.is_del())
          {
            // these nodes are dummy placeholders and don't actually have an associated
            // executed, so they are completed ... now.
            node_ids.push_back(out);
          } else {
            throw std::runtime_error("should account for all nodes in completed");
          }
        } else if(cnt < 0) {
          throw std::runtime_error("cnt should not be negative");
        }
      }
    }
  } // the mutex is released here

  cv.notify_all();
}

void cpu_mg_exec_state_t::apply_runner(int runner_id) {
  int which;
  bool is_first;
  while(true)
  {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&which, &is_first, this]() {
        if(num_remaining == 0) {
          return true;
        }
        for(auto iter = apply_ready.begin(); iter != apply_ready.end(); ++iter) {
          auto const& id = *iter;
          auto const& group_id = memgraph.nodes[id].op.get_apply().group;
          if(group_id < 0 || busy_groups.count(group_id) == 0) {
            which = id;

            if(group_id >= 0) {
              bool& f = is_firsts.at(group_id);
              is_first = f;
              f = false;
            } else {
              is_first = false;
            }

            apply_ready.erase(iter);
            busy_groups.insert(group_id);

            return true;
          }
        }
        return false;
      });

      if(num_remaining == 0) {
        return;
      }
    }

    auto const& apply = memgraph.nodes[which].op.get_apply();
    auto const& mems = apply.mems;
    if(apply.is_einsummable()) {
      auto const& e = apply.get_einsummable();

      vector<void const*> inns;
      inns.reserve(mems.size() - 1);
      for(int i = 1; i != mems.size(); ++i) {
        inns.push_back(get_raw(mems[i]));
      }

      auto [workspace, which_workspace] = workspace_manager.get(e);

      kernel_manager(e, get_raw(mems[0]), inns, workspace);

      workspace_manager.release(which_workspace);
    } else if(apply.is_touch()) {
      auto touch = apply.get_touch();
      if(is_first) {
        touch.castable = std::nullopt;
      }

      kernel_manager(touch, get_raw(mems[0]), get_raw(mems[1]));
    } else {
      throw std::runtime_error("missing apply op type impl");
    }

    completed(which, apply.group);
  }
}

void cpu_mg_exec_state_t::storage_runner(int runner_id) {
  if(runner_id > 0) {
    throw std::runtime_error("only one storage runner allowed");
  }
  if(!storage) {
    throw std::runtime_error("must have storage ptr");
  }

  storage_t& sto = *storage;

  int node_id;
  while(true)
  {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&node_id, this](){
        if(num_remaining == 0) {
          return true;
        }
        if(sto_ready.size() > 0) {
          node_id = sto_ready.front();
          sto_ready.pop();
          return true;
        }
        return false;
      });

      if(num_remaining == 0) {
        return;
      }
    }
    auto const& node = memgraph.nodes[node_id];
    if(!node.op.is_local_to(this_rank)) {
      throw std::runtime_error("node given to storage runner is not local");
    }

    if(node.op.is_evict()) {
      auto const& evict = node.op.get_evict();
      buffer_t data = get_buffer_reference(evict.src.as_mem());
      int tensor_id = evict.dst.id;
      sto.write(data, tensor_id);
    } else if(node.op.is_load()) {
      auto const& load = node.op.get_load();
      buffer_t data = get_buffer_reference(load.dst.as_mem());
      int tensor_id = load.src.id;
      sto.load(data, tensor_id);
    } else {
      throw std::runtime_error("storage runner must have evict or load node");
    }

    completed(node_id);
  }
}

void cpu_mg_exec_state_t::send_runner(int runner_id) {
  int send_id;

  while(true)
  {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&send_id, this](){
        if(num_remaining == 0) {
          return true;
        }
        if(send_ready.size() > 0) {
          send_id = send_ready.front();
          send_ready.pop();
          return true;
        }
        return false;
      });

      if(num_remaining == 0) {
        return;
      }
    }

    auto const& [src_info,dst_info,size] = memgraph.nodes[send_id].op.get_move();
    auto const& [_0 , offset_src] = src_info;
    auto const& [dst, _1        ] = dst_info;

    buffer_t data = get_buffer_reference(mem_t{ offset_src, size });

    mpi->send_int(send_id, dst, mpi->max_tag-1);
    mpi->send(data, dst, send_id);

    completed(send_id);
  }
}

void cpu_mg_exec_state_t::recv_runner(int runner_id) {
  int recv_id;
  int& n = num_recv_remaining;
  while(true) {
    {
      std::unique_lock lk(m);
      if(n == 0) {
        return;
      }
      n--;
    }

    recv_id = mpi->recv_int_from_anywhere(mpi->max_tag-1);

    auto const& [src_info,dst_info,size] = memgraph.nodes[recv_id].op.get_move();
    auto const& [src, _0        ] = src_info;
    auto const& [_1 , offset_dst] = dst_info;

    buffer_t data = get_buffer_reference(mem_t{ offset_dst, size });
    mpi->recv(data, src, recv_id);

    completed(recv_id);
  }
}

void cpu_mg_exec_state_t::recv_can_recv_runner() {
  int& n = num_recv_can_recv_remaining;
  for(; n != 0; n--) {
    int send_id = mpi->recv_int_from_anywhere(mpi->max_tag);

    bool should_notify = false;
    {
      std::unique_lock lk(m);
      auto iter = num_deps_remaining.find(send_id);
      int& cnt = iter->second;
      cnt--;
      if(cnt == 0) {
        num_deps_remaining.erase(iter);
        should_notify = true;
        send_ready.push(send_id);
      }
    }

    if(should_notify) {
      cv.notify_all();
    }
  }
}

void cpu_mg_exec_state_t::send_can_recv_runner() {
  int recv_id;
  while(true)
  {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&recv_id, this](){
        if(num_remaining == 0) {
          return true;
        }
        if(can_recv_ready.size() > 0) {
          recv_id = can_recv_ready.front();
          can_recv_ready.pop();
          return true;
        }
        return false;
      });

      if(num_remaining == 0) {
        return;
      }
    }

    auto const& move = memgraph.nodes[recv_id].op.get_move();
    int const& src = move.get_src_loc();
    int const& dst = move.get_dst_loc();
    mpi->send_int(recv_id, move.get_src_loc(), mpi->max_tag);
  }
}

////////////////////////

_tg_with_mg_helper_t::_tg_with_mg_helper_t(
  mpi_t* a,
  map<int, memsto_t>& b,
  buffer_t& c,
  storage_t& d,
  int e)
  : mpi(a), data_locs(b), mem(c), storage(d), server_rank(e)
{
  if(mpi) {
    this_rank  = mpi->this_rank;
    world_size = mpi->world_size;
  } else {
    this_rank = 0;
    world_size = 1;
    if(server_rank != 0) {
      throw std::runtime_error("invalid server rank");
    }
  }
}

vector<uint64_t> _tg_with_mg_helper_t::recv_mem_sizes() {
  vector<uint64_t> mem_sizes;
  mem_sizes.reserve(world_size);
  for(int src = 0; src != world_size; ++src) {
    if(src == this_rank) {
      mem_sizes.push_back(mem->size);
    } else {
      vector<uint64_t> singleton = mpi->recv_vector<uint64_t>(src);
      mem_sizes.push_back(singleton[0]);
    }
  }
  return mem_sizes;
}

void _tg_with_mg_helper_t::send_mem_size() {
  vector<uint64_t> singleton{mem->size};
  mpi->send_vector(singleton, server_rank);
}

map<int, memstoloc_t> _tg_with_mg_helper_t::recv_full_data_locs() {
  auto fix = [](memsto_t const& memsto, int loc) {
    if(memsto.is_mem()) {
      return memstoloc_t(memsto.get_mem().as_memloc(loc));
    } else {
      int const& sto_id = memsto.get_sto();
      return memstoloc_t(stoloc_t { loc, sto_id });
    }
  };

  map<int, memstoloc_t> ret;
  for(int src = 0; src != world_size; ++src) {
    if(src == this_rank) {
      for(auto const& [id, memsto]: data_locs) {
        ret.insert({id, fix(memsto, src)});
      }
    } else {
      auto src_data_locs = mpi->recv_vector<id_memsto_t>(src);
      for(auto const& [id, memsto]: src_data_locs) {
        ret.insert({id, fix(memsto, src)});
      }
    }
  }

  return ret;
}

void _tg_with_mg_helper_t::send_data_locs() {
  vector<id_memsto_t> items;
  for(auto const& [id, memsto]: data_locs) {
    items.push_back(id_memsto_t { id, memsto });
  }
  mpi->send_vector(items, server_rank);
}

void _tg_with_mg_helper_t::storage_remap_server(
  vector<vector<std::array<int, 2>>> const& storage_remaps)
{
  for(int dst = 0; dst != world_size; ++dst) {
    if(dst == this_rank) {
      storage.remap(storage_remaps[this_rank]);
    } else {
      mpi->send_vector(storage_remaps[dst], dst);
    }
  }
}

void _tg_with_mg_helper_t::storage_remap_client() {
  auto storage_remap = mpi->recv_vector<std::array<int, 2>>(server_rank);
  storage.remap(storage_remap);
}

void _tg_with_mg_helper_t::broadcast_memgraph(memgraph_t const& mg) {
  for(int dst = 0; dst != world_size; ++dst) {
    if(dst != this_rank) {
      mpi->send_str(mg.to_wire(), dst);
    }
  }
}

memgraph_t _tg_with_mg_helper_t::recv_memgraph() {
  return memgraph_t::from_wire(mpi->recv_str(server_rank));
}

void _tg_with_mg_helper_t::rewrite_data_locs_server(
  map<int, memstoloc_t> const& new_data_locs)
{
  auto get_loc = [](memstoloc_t const& x) {
    if(x.is_memloc()) {
      return x.get_memloc().loc;
    } else {
      // Note: stoloc_t::loc is a storage location, but we are
      //       assuming that storage loc == compute loc in this directory!
      return x.get_stoloc().loc;
    }
  };

  vector<vector<id_memsto_t>> items;
  data_locs.clear();
  for(auto const& [id, memstoloc]: new_data_locs) {
    int loc = get_loc(memstoloc);
    if(loc == this_rank) {
      data_locs.insert({ id, memstoloc.as_memsto() });
    } else {
      items[loc].push_back(id_memsto_t { id, memstoloc.as_memsto() });
    }
  }

  for(int dst = 0; dst != world_size; ++dst) {
    if(dst != this_rank) {
      mpi->send_vector(items[dst], dst);
    }
  }
}

void _tg_with_mg_helper_t::rewrite_data_locs_client() {
  data_locs.clear();
  auto new_data_locs = mpi->recv_vector<id_memsto_t>(server_rank);
  for(auto const& [id, memsto]: new_data_locs) {
    data_locs.insert({id, memsto});
  }
}

vector<vector<std::array<int, 2>>>
_tg_with_mg_helper_t::create_storage_remaps(
  map<int, memstoloc_t> const& full_data_locs,
  map<int, memstoloc_t> const& inn_tg_to_loc)
{
  vector<vector<std::array<int, 2>>> storage_remaps(world_size);
  for(auto const& [id, mg_memstoloc]: inn_tg_to_loc) {
    if(mg_memstoloc.is_stoloc()) {
      auto const& [loc, new_sto_id] = mg_memstoloc.get_stoloc();
      auto const& [_, old_sto_id] = full_data_locs.at(id).get_stoloc();
      storage_remaps[loc].push_back({new_sto_id, old_sto_id});
    }
  }
  return storage_remaps;
}

memgraph_t execute_taskgraph_as_memgraph_server(
  taskgraph_t const& taskgraph,
  execute_memgraph_settings_t const& exec_settings,
  kernel_manager_t const& kernel_manager,
  allocator_settings_t const& alloc_settings,
  mpi_t* mpi,
  map<int, memsto_t>& data_locs,
  buffer_t mem,
  storage_t& storage)
{
  int this_rank  = bool(mpi) ? 0 : mpi->this_rank;
  _tg_with_mg_helper_t helper(mpi, data_locs, mem, storage, this_rank);

  vector<uint64_t> mem_sizes = helper.recv_mem_sizes();

  map<int, memstoloc_t> full_data_locs = helper.recv_full_data_locs();

  vector<int> which_storage(helper.world_size);
  std::iota(which_storage.begin(), which_storage.end(), 0);

  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
    memgraph_t::make(
      taskgraph, which_storage, mem_sizes,
      full_data_locs, alloc_settings, true);

  // memgraph now uses wtvr storage ids it chooses... So for each input,
  // figure out what the remap is
  vector<vector<std::array<int, 2>>> storage_remaps =
    helper.create_storage_remaps(full_data_locs, inn_tg_to_loc);

  // this is not need anymore
  full_data_locs.clear();

  helper.storage_remap_server(storage_remaps);

  helper.broadcast_memgraph(memgraph);

  execute_memgraph(
    memgraph, exec_settings, kernel_manager,
    mpi, mem, storage);

  helper.rewrite_data_locs_server(out_tg_to_loc);

  return memgraph;
}

void execute_taskgraph_as_memgraph_client(
  execute_memgraph_settings_t const& exec_settings,
  kernel_manager_t const& kernel_manager,
  int server_rank,
  mpi_t* mpi,
  map<int, memsto_t>& data_locs,
  buffer_t mem,
  storage_t& storage)
{
  _tg_with_mg_helper_t helper(mpi, data_locs, mem, storage, server_rank);

  helper.send_mem_size();

  helper.send_data_locs();

  helper.storage_remap_client();

  memgraph_t memgraph = helper.recv_memgraph();

  execute_memgraph(
    memgraph, exec_settings, kernel_manager,
    mpi, mem, storage);

  helper.rewrite_data_locs_client();
}

