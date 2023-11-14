#include "../src/engine/cpu/kernel_executor.h"
#include "../src/einsummable/dbuffer.h"

struct datum_t {
  einsummable_t e;
  vector<double> times;

  static std::ostream& header(std::ostream& out);
};

std::ostream& operator<<(std::ostream& out, datum_t const& ts);

vector<einsummable_t> make_experiment_set();

datum_t run(cpu_kernel_executor_t& executor, einsummable_t const& e, int nrep);

int main(int argc, char** argv) {
  cpu_kernel_executor_t executor;

  datum_t::header(std::cout) << std::endl;

  int nrep = 10;
  for(einsummable_t const& e: make_experiment_set()) {
    datum_t d = run(executor, e, nrep);
    std::cout << d << std::endl;
  }
}

std::ostream& datum_t::header(std::ostream& out) {
  out << "str, lhs_shape, rhs_shape, out_shape, times...";
  return out;
}

std::ostream& operator<<(std::ostream& out, datum_t const& d) {
  auto inn_shapes = d.e.inn_shapes();
  out << d.e.str() 
    << ", " << inn_shapes[0] 
    << ", " << inn_shapes[1] 
    << ", " << d.e.out_shape();
  for(double const& t: d.times) {
    out << ", " << t;
  }
  return out;
}

datum_t run(cpu_kernel_executor_t& executor, einsummable_t const& e, int nrep) {
  uint64_t workspace_size = executor.build(e).value(); 
  vector<uint64_t> workspace_vector(workspace_size);
  tuple<void*, uint64_t> workspace ( 
    reinterpret_cast<void*>(workspace_vector.data()), 
    workspace_size);

  vector<dbuffer_t> inn_dbuffers;
  vector<void const*> inns;
  auto inn_shapes = e.inn_shapes();
  auto inn_dtypes = e.inn_dtypes();
  for(int i = 0; i != inn_shapes.size(); ++i) {
    auto const& shape = inn_shapes[i];
    auto const& dtype = inn_dtypes[i];
    dbuffer_t dbuffer = make_dbuffer(dtype, product(shape));
    dbuffer.random("-0.000001", "0.000001");
    inn_dbuffers.push_back(dbuffer);
    inns.push_back(dbuffer.raw());
  }

  dbuffer_t out_dbuffer = make_dbuffer(e.out_dtype(), product(e.out_shape()));
  void* out = out_dbuffer.raw();

  datum_t ret {
    .e = e,
    .times = {}
  };
  for(int i = 0; i != nrep; ++i) {
    auto start = clock_now();
    executor(e, out, inns, workspace);
    auto end = clock_now();
    double time = std::chrono::duration<double>(end-start).count();
    ret.times.push_back(time);
  }

  return ret;
}

vector<einsummable_t> _make_es(
  string str,
  vector<uint64_t> join_shape,
  vector<vector<int>> splits)
{
  vector<einsummable_t> ret;
  auto [inns, out_rank] = einsummable_t::parse_str(str);
  auto mul = scalarop_t::make_mul(dtype_t::f32);
  for(auto const& split: splits) {
    vector<uint64_t> small_join_shape = join_shape;
    for(int i = 0; i != join_shape.size(); ++i) {
      small_join_shape[i] /= split[i];
    }
    ret.emplace_back(small_join_shape, inns, out_rank, mul, castable_t::add);
  }
  return ret;
}

vector<einsummable_t> make_experiment_set() {
  vector<einsummable_t> ret;

  vector_concatenate_into(ret, _make_es(
    "abef,cdef->abcd",
    vector<uint64_t>{1,512,64,128,64,128},
    vector<vector<int>> {
      {1,1,2,1,16,1},
      {1,2,16,1,1,1},
      {1,2,32,1,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "acbe,adbe->abcd",
    vector<uint64_t>{1,64,512,512,128},
    vector<vector<int>> {
      {1,2,1,16,1},
      {1,32,2,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "abce,aebd->abcd",
    vector<uint64_t>{1,64,512,128,512},
    vector<vector<int>> {
      {1,1,1,1,32},
      {1,32,2,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "aebf,cdef->abcd",
    vector<uint64_t>{1,512,64,128,64,128},
    vector<vector<int>> {
      {1,1,32,1,1,1},
      {1,2,32,1,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "abef,cdef->abcd",
    vector<uint64_t>{1,512,32,128,32,128},
    vector<vector<int>> {
      {1,2,16,1,1,1},
      {1,2,32,1,1,1},
      {1,1,1,1,2,1},
      {1,2,1,1,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "acbe,adbe->abcd",
    vector<uint64_t>{1,32,512,512,128},
    vector<vector<int>> {
      {1,32,2,1,1},
      {1,1,2,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "abce,aebd->abcd",
    vector<uint64_t>{1,32,512,128,512},
    vector<vector<int>> {
      {1,32,2,1,1},
      {1,1,2,1,1}
    }
  ));

  vector_concatenate_into(ret, _make_es(
    "aebf,cdef->abcd",
    vector<uint64_t>{1,512,32,128,32,128},
    vector<vector<int>> {
      {1,2,16,1,1,1},
      {1,2,32,1,1,1},
      {1,2,1,1,1,1}
    }
  ));

  for(int i = 0; i != ret.size()-1; ++i) {
    for(int j = i+1; j != ret.size(); ++j) {
      auto const& e1 = ret[i];
      auto const& e2 = ret[j];
      if(e1.join_shape == e2.join_shape && e1.str() == e2.str()) {
        throw std::runtime_error("there are duplicates! see " + write_with_ss(e1));
      }
    }
  }

  return ret;
}

