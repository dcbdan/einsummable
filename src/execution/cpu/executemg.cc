#include "executemg.h"
#include "workspace.h"

#include <thread>
#include <mutex>
#include <condition_variable>

using std::thread;
using std::queue;

#define RLINEOUT(x) // if(mpi->this_rank == 0) { DLINEOUT(x); }

struct cpu_mg_exec_state_t {
  cpu_mg_exec_state_t(
    mpi_t* mpi,
    memgraph_t const& memgraph,
    kernel_manager_t const& kernel_manager,
    buffer_t& buffer);

  void apply_runner(int runner_id);
  void cache_runner(int runner_id);
  void send_runner(int runner_id);
  void recv_runner(int runner_id);

  // these should be launched just once
  void send_can_recv_runner();
  void recv_can_recv_runner();

  void completed(int node_id, int group_id = -1);

  void run(int n_apply, int n_cache, int n_send, int n_recv);

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

  queue<int> cache_ready;

  int num_remaining;

  map<int, int> num_deps_remaining;

  int num_recv_remaining;
  int num_recv_can_recv_remaining;

  set<int> busy_groups;
};

void execute_memgraph(
  memgraph_t const& memgraph,
  execute_memgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi,
  buffer_t memory)
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

  cpu_mg_exec_state_t state(mpi, memgraph, kernel_manager, memory);

  int world_size = bool(mpi) ? mpi->world_size : 1;

  state.run(
    settings.num_apply_runner,
    settings.num_cache_runner,
    world_size > 1 ? settings.num_send_runner : 0,
    world_size > 1 ? settings.num_recv_runner : 0);
}

void cpu_mg_exec_state_t::run(int n_apply, int n_cache, int n_send, int n_recv)
{
  vector<thread> runners;
  runners.reserve(n_apply + n_cache + n_send + n_recv + 2);
  for(int i = 0; i != n_apply; ++i) {
    runners.emplace_back([this, i](){ return this->apply_runner(i); });
  }
  for(int i = 0; i != n_cache; ++i) {
    runners.emplace_back([this, i](){ return this->cache_runner(i); });
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
  buffer_t& b)
  : mpi(mpi),
    memgraph(mg),
    kernel_manager(km),
    buffer(b),
    workspace_manager(km)
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
            cache_ready.push(out);
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
  while(true)
  {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&which, this]() {
        if(num_remaining == 0) {
          return true;
        }
        for(auto iter = apply_ready.begin(); iter != apply_ready.end(); ++iter) {
          auto const& id = *iter;
          auto const& group_id = memgraph.nodes[id].op.get_apply().group;
          if(group_id < 0 || busy_groups.count(group_id) == 0) {
            which = id;
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
      auto const& touch = apply.get_touch();
      kernel_manager(touch, get_raw(mems[0]), get_raw(mems[1]));
    } else {
      throw std::runtime_error("missing apply op type impl");
    }

    completed(which, apply.group);
  }
}

void cpu_mg_exec_state_t::cache_runner(int runner_id) {
  throw std::runtime_error("cache_runner not implemented");
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