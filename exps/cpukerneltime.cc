#include "../src/einsummable/einsummable.h"
#include "../src/execution/cpu/kernels.h"
#include "../src/einsummable/dbuffer.h"

#include <fstream>

#include <thread>
#include <mutex>
#include <condition_variable>

struct runner_state_t {
  runner_state_t() {}

  void register_einsummable(einsummable_t const& e) {
    if(!kernel_manager.build(e)) {
      throw std::runtime_error("could not build this einsummable");
    }

    int out_id = make_data(e.out_dtype(), e.out_shape());

    vector<int> inn_ids;

    auto inn_shapes = e.inn_shapes();
    inn_ids.reserve(inn_shapes.size());
    for(int i = 0; i != inn_shapes.size(); ++i) {
      inn_ids.push_back(
        make_data(e.inn_dtype(i), inn_shapes[i]));
    }

    ops.push_back(op_t {
      .e = e,
      .out = out_id,
      .inns = inn_ids
    });
  }

  struct op_t {
    einsummable_t e;
    int out;
    vector<int> inns;
  };

  struct datum_t {
    uint64_t inn_elem;
    uint64_t out_elem;
    uint64_t compute;
    double time;
  };

  void runner(int runner_id) {
    int id;
    while(true) {
      {
        std::unique_lock lk(m_apply);
        cv_apply.wait(lk, [&, this] {
          id = which_op++;
          return true;
        });

        if(id >= ops.size()) {
          return;
        }
      }

      auto const& [e, out_id ,inn_ids] = ops[id];

      double wall_time;
      {
        vector<void const*> inn_ptrs;
        inn_ptrs.reserve(inn_ids.size());
        for(auto const& inn_id: inn_ids) {
          inn_ptrs.push_back(data[inn_id]->raw());
        }

        uint64_t workspace_size = kernel_manager.workspace_size(e);
        buffer_t bws;
        optional<tuple<void*, uint64_t>> workspace;
        if(workspace_size > 0) {
          bws = make_buffer(workspace_size);
          workspace = tuple<void*, uint64_t>(bws->raw(), workspace_size);
        }

        timestamp_t start = clock_now();
        kernel_manager(e, data[out_id]->raw(), inn_ptrs, workspace);
        timestamp_t end = clock_now();
        wall_time = std::chrono::duration<
          double, std::milli>(end-start).count();
      }

      uint64_t inn_total = 0;
      for(auto const& shape: e.inn_shapes()) {
        inn_total += product(shape);
      }

      datum_t datum {
        .inn_elem = inn_total,
        .out_elem = e.out_nelem(),
        .compute = product(e.join_shape),
        .time = wall_time
      };

      std::unique_lock lk(m_stats);
      stats.push_back(datum);
    }
  }

  void launch(int num_threads) {
    which_op = 0;

    vector<std::thread> runners;
    runners.reserve(num_threads);
    for(int i = 0; i != num_threads; ++i) {
      runners.emplace_back([this, i](){ return this->runner(i); });
    }

    for(auto& t: runners) {
      t.join();
    }
  }

  int make_data(dtype_t dtype, vector<uint64_t> const& shape) {
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    d.random("-0.001", "0.001");
    data.push_back(d.data);
    return data.size()-1;
  }

  void print_csv(std::ostream& out) const {
    out << "inn_elem,out_elem,compute,time(ms)" << std::endl;
    for(auto const& d: stats) {
      out << d.inn_elem << "," << d.out_elem
          << "," << d.compute << "," << d.time
          << std::endl;
    }
  }

  vector<op_t> ops;

  vector<buffer_t> data;
  kernel_manager_t kernel_manager;

  int which_op;

  std::mutex m_apply;
  std::condition_variable cv_apply;

  std::mutex m_stats;
  vector<datum_t> stats;
};

void test_matmuls() {
  vector<uint64_t> possible_sizes = { 32, 8, 64, 1024, 2048 };
  int num_matmuls = 4000;
  int num_threads = 2;

  runner_state_t ss;

  set_seed(0);
  auto random_size = [&] {
    return possible_sizes[runif(possible_sizes.size())];
  };

  for(int _ = 0; _ != num_matmuls; ++_) {
    uint64_t i = random_size();
    uint64_t j = random_size();
    uint64_t k = random_size();
    ss.register_einsummable(einsummable_t::from_matmul(i,j,k));
  }

  std::cout << "Launching..." << std::endl;
  ss.launch(num_threads);

  string filename = "test_matmuls_nt" + write_with_ss(num_threads) + ".csv";

  std::ofstream f(filename);
  ss.print_csv(f);
  std::cout << "Printed to " << filename << std::endl;
}

void test_ewbs(int num_threads) {
  vector<uint64_t> possible_sizes = { 32, 8, 64, 1024, 2048 };
  int num_ops = 2000;

  runner_state_t ss;

  set_seed(0);
  auto random_size = [&] {
    return possible_sizes[runif(possible_sizes.size())];
  };

  for(int _ = 0; _ != num_ops; ++_) {
    uint64_t i = random_size();
    uint64_t j = random_size();
    ss.register_einsummable(einsummable_t(
      {i,j},
      { {0,1}, {0,1} },
      2,
      scalarop_t::make_mul()
    ));
  }

  std::cout << "Launching..." << std::endl;
  ss.launch(num_threads);

  string filename = "test_ewbs_nt" + write_with_ss(num_threads) + ".csv";

  std::ofstream f(filename);
  ss.print_csv(f);
  std::cout << "Printed to " << filename << std::endl;
}

int main() {
  test_ewbs(1);
  test_ewbs(2);
  test_ewbs(4);
}
