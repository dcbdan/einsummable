#pragma once
#include "../base/setup.h"

#include "resource_manager.h"
#include "threadpool.h"

struct global_buffers_t
  : rm_template_t<int, void*>
{
  global_buffers_t(void* ptr)
    : global_buffers_t(vector<void*>{ ptr })
  {}

  global_buffers_t(vector<void*> ps):
    ptrs(ps)
  {}

  static desc_ptr_t make_desc() { return rm_template_t::make_desc(0); }

  static desc_ptr_t make_desc(int device){
    return rm_template_t::make_desc(device);
  }

  static desc_ptr_t make_multi_desc(std::vector<int> const& whichs) {
    vector<desc_ptr_t> ret;
    for(auto which: whichs) {
      ret.push_back(rm_template_t::make_desc(which));
    }
    return resource_manager_t::make_desc(ret);
  }

private:
  optional<void*> try_to_acquire_impl(int const& which) {
    return ptrs.at(which);
  }

  void release_impl(void* const& ptr) {}

  vector<void*> ptrs;
};

struct group_manager_t
  : rm_template_t<int, tuple<int, bool>>
{
  group_manager_t() {}

private:
  optional<tuple<int, bool>> try_to_acquire_impl(int const& group_id);

  void release_impl(tuple<int, bool> const& group_id_and_is_first);

  std::mutex m;
  set<int> busy_groups;
  set<int> seen_groups;
};

struct threadpool_manager_t;

struct threadpool_resource_t {
  threadpool_resource_t(int i, threadpool_manager_t* s)
    : id(i), self(s)
  {}

  void launch(std::function<void()> f) const { launch("tp_resource_na", f); }
  void launch(string label, std::function<void()> f) const;

private:
  friend class threadpool_manager_t;

  int id;
  threadpool_manager_t* self;
};

struct threadpool_manager_t
  : rm_template_t<unit_t, threadpool_resource_t>
{
  threadpool_manager_t(threadpool_t& tp);

  static desc_ptr_t make_desc() {
    return rm_template_t::make_desc(unit_t {});
  }

private:
  optional<threadpool_resource_t> try_to_acquire_impl(unit_t const&);

  // assumption: the corresponding function that got launched is complete
  //             when this is called
  void release_impl(threadpool_resource_t const&);

  int num_avail;
  threadpool_t& threadpool;

  std::mutex m;

  int id_;
  set<int> was_called;
private:
  friend class threadpool_resource_t;

  void launch(int which, string label, std::function<void()> f);
};
