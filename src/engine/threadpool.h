#pragma once
#include "../base/setup.h"

#include <thread>
#include <mutex>
#include <condition_variable>

#include <fstream>

struct threadpool_t {
  threadpool_t(int num_to_launch, bool pin=false)
    : threadpool_t("", num_to_launch, pin)
  {}

  threadpool_t(string filename, int num_to_launch, bool pin=false);

  ~threadpool_t();

  void insert(std::function<void()> f) { insert("na", f); }
  void insert(string label, std::function<void()> f);

  int num_runners() const { return threads.size(); }

private:
  std::mutex m;
  std::condition_variable cv;

  std::queue<tuple<string, std::function<void()>>> work_queue;
  bool is_stopped;

  vector<std::thread> threads;
  timestamp_t const start_threadpool;
  std::mutex m_print;

  optional<std::ofstream> out;

private:
  void runner(int which, bool pin = false);

  void print_time(
    int which, string label, 
    timestamp_t const& start, timestamp_t const& end);
};

