#include "threadpool.h"
#include "numa.h"

#include <sys/sysinfo.h>

threadpool_t::threadpool_t(int num_to_launch)
  : is_stopped(false), start(clock_now())
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
  get_numa_info().pin_to_thread(which);

  std::function<void()> f;

  auto check_and_set = [&, this] {
    if(is_stopped) {
      return true;
    }
    if(work_queue.size() > 0) {
      f = work_queue.front();
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

    print_time(which);
    f();
    print_time(which);
  }
}

void threadpool_t::insert(std::function<void()> f) {
  print_time(-1);
  {
    std::unique_lock lk(m);

    if(is_stopped) {
      throw std::runtime_error("this thread pool has been shutdown");
    }

    work_queue.push(f);
  }
  print_time(-1);

  cv.notify_one();
}

void threadpool_t::print_time(int which) {
  auto end = clock_now();
  using namespace std::chrono;
  auto duration = (double) duration_cast<microseconds>(end - start).count()
                / (double) duration_cast<microseconds>(1s         ).count();
  //std::unique_lock lk(m_print);
  //DOUT(which << " " << duration);
}

