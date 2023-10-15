#include "threadpool.h"

#include <sched.h>
#include <sys/sysinfo.h>

#include "numa.h"

std::mutex threadpool_total_mutex;
double threadpool_total = 0.0;

ghost_t make_threadpool_ghost() {
  return ghost_t(threadpool_total_mutex, threadpool_total);
}

double get_threadpool_total() { return threadpool_total; }

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

  //get_numa_info().pin_to_this_numa_thread(which);
 
  //{
  //  cpu_set_t cpu_set;
  //  CPU_ZERO(&cpu_set);
  //  CPU_SET(which, &cpu_set);
  //  if(sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
  //    throw std::runtime_error("could not pin thread");
  //  }
  //}

  std::function<void()> f;

  while(true) {
    {
      std::unique_lock lk(m);
      cv.wait(lk, [&, this] {
        if(is_stopped) {
          return true;
        }
        if(work_queue.size() > 0) {
          f = work_queue.front();
          work_queue.pop();
          return true;
        }
        return false;
      });

      if(is_stopped) {
        return;
      }
    }

    ghost_t ghost = make_threadpool_ghost();
    //{
    //  std::unique_lock lk(threadpool_total_mutex);
    //  auto end = clock_now();
    //  using namespace std::chrono;
    //  auto duration = (double) duration_cast<microseconds>(end - start).count()
    //                / (double) duration_cast<microseconds>(1s         ).count();
    //  DOUT(which << " " << duration);
    //}
    f();
    //{
    //  std::unique_lock lk(threadpool_total_mutex);
    //  auto end = clock_now();
    //  using namespace std::chrono;
    //  auto duration = (double) duration_cast<microseconds>(end - start).count()
    //                / (double) duration_cast<microseconds>(1s         ).count();
    //  DOUT(which << " " << duration);
    //}
  }
}

void threadpool_t::insert(std::function<void()> f) {
  {
    std::unique_lock lk(m);

    if(is_stopped) {
      throw std::runtime_error("this thread pool has been shutdown");
    }

    work_queue.push(f);
  }

  cv.notify_one();
}
