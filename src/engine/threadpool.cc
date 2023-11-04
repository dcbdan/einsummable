#include "threadpool.h"
#include "numa.h"

#include <sys/sysinfo.h>

threadpool_t::threadpool_t(string filename, int num_to_launch)
  : is_stopped(false), start_threadpool(clock_now()), out(filename)
{
  get_numa_info();

  int max_num_threads = get_nprocs();

  // Either don't allow num_to_launch > max_num_threads
  // or make sure a runner has id < max_num_threads

  threads.reserve(num_to_launch);
  for(int i = 0; i != num_to_launch; ++i) {
    threads.emplace_back([this, i, max_num_threads] {
      this->runner(i % max_num_threads);
    });
  }
}

threadpool_t::~threadpool_t() {
  {
    std::unique_lock lk(m);

    if(is_stopped) {
      return;
    }

    is_stopped = true;
  }

  cv.notify_all();

  for(auto& t: threads) {
    t.join();
  }
}

void threadpool_t::runner(int which) {
  //get_numa_info().pin_to_this_numa_thread(which);
  get_numa_info().pin_to_thread(which);

  string label;
  std::function<void()> f;

  auto check_and_set = [&, this] {
    if(is_stopped) {
      return true;
    }
    if(work_queue.size() > 0) {
      auto const& [l_,f_] = work_queue.front();
      label = l_;
      f = f_;
      work_queue.pop();
      return true;
    }
    return false;
  };

  while(true) {
    {
      std::unique_lock lk(m);

      if(!check_and_set()) {
        cv.wait(lk, check_and_set);
      }

      if(is_stopped) {
        return;
      }
    }

    auto start = clock_now();
    f();
    auto end = clock_now();
    print_time(which, label, start, end);
  }
}

void threadpool_t::insert(string label, std::function<void()> f) {
  {
    std::unique_lock lk(m);

    if(is_stopped) {
      throw std::runtime_error("this thread pool has been shutdown");
    }

    work_queue.push({label, f});
  }

  cv.notify_one();
}

void threadpool_t::print_time(
   int which, string label, 
   timestamp_t const& start, timestamp_t const& end) 
{
  using namespace std::chrono;
  double s = duration<double>(start - start_threadpool).count();
  double e = duration<double>(end   - start_threadpool).count();
  std::unique_lock lk(m_print);
  out << which << " " << label << " " << s << " " << e << std::endl;
}

