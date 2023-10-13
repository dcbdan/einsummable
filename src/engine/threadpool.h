#pragma once
#include "../base/setup.h"

#include <thread>
#include <mutex>
#include <condition_variable>

struct threadpool_t {
  threadpool_t(int num_to_launch);

  ~threadpool_t();

  void insert(std::function<void()> f);

  int num_runners() const { return threads.size(); }

private:
  std::mutex m;
  std::condition_variable cv;

  std::queue<std::function<void()>> work_queue;
  bool is_stopped;

  vector<std::thread> threads;

private:
  void runner(int which);
};

