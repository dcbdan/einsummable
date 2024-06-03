#include "setup.h"

#include <mutex>

struct timetracker_t {
  timetracker_t();

  using timestamp_t =
    typename std::chrono::time_point<std::chrono::steady_clock>;
  using duration_t =
    typename std::chrono::duration<double>;
  using interval_t = tuple<timestamp_t, timestamp_t>;

  struct gremlin_t {
    gremlin_t(timetracker_t& self, string name_, bool is_interval, bool is_total, uint64_t f = 0);

    ~gremlin_t();

    timetracker_t& self;
    string name;
    bool is_interval;
    bool is_total;
    uint64_t flops;
    timestamp_t start;
  };

  static timestamp_t now();

  void insert_full(string name, timestamp_t start, timestamp_t end, uint64_t f = 0);
  void insert_interval(string name, timestamp_t start, timestamp_t end, uint64_t f = 0);
  void insert_total(string name, timestamp_t start, timestamp_t end, uint64_t f = 0);

  gremlin_t make_full_gremlin(string name, uint64_t f = 0);
  gremlin_t make_interval_gremlin(string name, uint64_t f = 0);
  gremlin_t make_totals_gremlin(string name, uint64_t f = 0);

  void print_intervals(std::ostream& out);
  void print_totals(std::ostream& out);

  void clear();
private:
  timestamp_t const world_start;

  std::mutex m;

  map<string, int> counts;
  map<string, uint64_t> flops;
  map<string, duration_t> totals;
  map<string, vector<interval_t>> intervals;

  void _insert_interval(string name, timestamp_t start, timestamp_t end);
  void _insert_total(string name, timestamp_t start, timestamp_t end);
};

timetracker_t& get_timetracker();

