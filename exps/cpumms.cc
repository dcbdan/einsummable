#include "../src/einsummable/graph.h"
#include "../src/execution/cpu/executetg.h"

#include <fstream>

#include <mkl_cblas.h>
#include <mkl.h>

tuple<taskgraph_t, map<int, buffer_t>>
make_tg_and_data(uint64_t sz_i, uint64_t sz_jk, int n, int npart_i, int npart_jk)
{
  partdim_t pd_i = partdim_t::split(sz_i,  npart_i);
  partdim_t pd_jk = partdim_t::split(sz_jk, npart_jk);
  partition_t part_l(vector<partdim_t>{pd_i, pd_jk});
  partition_t part_r(vector<partdim_t>{pd_jk, pd_jk});
  partition_t part_j(vector<partdim_t>{pd_i, pd_jk, pd_jk});

  einsummable_t einsummable = einsummable_t::from_matmul(sz_i, sz_jk, sz_jk);

  graph_constructor_t gc;
  {
    vector<int> tensors;
    tensors.push_back(gc.insert_input(part_l));
    for(int i = 0; i != n; ++i) {
      tensors.push_back(gc.insert_input(part_r));
    }

    for(int i = 1; i != n+1; ++i) {
      tensors[0] = gc.insert_formation(
        part_l,
        gc.insert_einsummable(
          part_j,
          einsummable,
          {tensors[0], tensors[i]}));
    }
  }

  auto pls = gc.get_placements();

  auto[gid_to_blocks, _, tg] = taskgraph_t::make(gc.graph, pls);

  map<int, buffer_t> data;
  for(auto const& [gid, tids_]: gid_to_blocks) {
    dtype_t dtype = gc.graph.nodes[gid].op.out_dtype();
    vector<int> const& tids = tids_.get();
    vector<uint64_t> block_sizes = pls[gid].partition.all_block_sizes().get();
    for(int blk = 0; blk != tids.size(); ++blk) {
      int const& tid        = tids[blk];
      uint64_t const& nelem = block_sizes[blk];
      dbuffer_t d = make_dbuffer(dtype, nelem);
      d.random("-0.10", "0.10");
      data.insert({tid, d.data});
    }
  }

  {
    std::ofstream f("g.gv");
    gc.graph.print_graphviz(f);
    std::cout << "printed graph_t to g.gv" << std::endl;
  }

  {
    std::ofstream f("tg.gv");
    tg.print_graphviz(f);
    std::cout << "printed taskgraph_t to tg.gv" << std::endl;
  }

  return {tg, data};
}

tuple<taskgraph_t, map<int, buffer_t>>
make_tg_and_data(uint64_t sz, int n, int npart) {
  return make_tg_and_data(sz, sz, n, npart, npart);
}

int test01(int argc, char** argv) {
  if(argc != 5) {
    std::cout << "Usage: sz n npart nrunner" << std::endl;
    return 1;
  }

  uint64_t sz = parse_with_ss<uint64_t>(argv[1]);
  int n       = parse_with_ss<int>(argv[2]);

  vector<dbuffer_t> data;
  for(int i = 0; i != n+1; ++i) {
    data.push_back(make_dbuffer(default_dtype(), sz*sz));
    data.back().random("-0.01", "0.01");
  }

  for(int i = 1; i != n+1; ++i) {
    dbuffer_t out = make_dbuffer(default_dtype(), sz*sz);
    gremlin_t gremlin("mm");
    matrix_multiply(
      default_dtype(),
      sz, sz, sz,
      false, false,
      out.raw(), data[0].raw(), data[i].raw());
    data[0] = out;
  }

  return 0;
}

int test02(int argc, char** argv) {
  if(argc != 5) {
    std::cout << "Usage: sz n npart nrunner" << std::endl;
    return 1;
  }

  uint64_t sz = parse_with_ss<uint64_t>(argv[1]);
  int n       = parse_with_ss<int>(argv[2]);

  vector<dbuffer_t> data;
  for(int i = 0; i != n+1; ++i) {
    data.push_back(make_dbuffer(default_dtype(), sz*sz));
    data.back().random("-0.1", "0.1");
  }

  uint64_t ni = sz;
  uint64_t nj = sz;
  uint64_t nk = sz;

  void* handle;
  auto status = mkl_jit_create_sgemm(
    &handle,
    MKL_ROW_MAJOR,
    MKL_NOTRANS, MKL_NOTRANS,
    ni, nk, nj,
    1.0, nj, nk, 0.0, nk);
  if(status == MKL_JIT_ERROR) {
    DOUT("JIT ERROR");
  } else if(status == MKL_NO_JIT) {
    DOUT("MKL NO JIT");
  } else if(status == MKL_JIT_SUCCESS) {
    DOUT("MKL JIT SUCCESS");
  } else {
    DOUT("??????????? UNKNOWN STATUS");
  }
  auto f = mkl_jit_get_sgemm_ptr(handle);

  for(int i = 1; i != n+1; ++i) {
    dbuffer_t out = make_dbuffer(default_dtype(), sz*sz);
    gremlin_t gremlin("mm");
    f(handle,
      reinterpret_cast<float*>(data[0].raw()),
      reinterpret_cast<float*>(data[i].raw()),
      reinterpret_cast<float*>(out.raw()));
    data[0] = out;
  }

  // TODO: gotta free handle

  return 0;
}

int test03(int argc, char** argv) {
  if(argc != 7) {
    std::cout << "Usage: sz_i sz_jk n npart_i npart_jk nrunner" << std::endl;
    return 1;
  }

  uint64_t sz_i  = parse_with_ss<uint64_t>(argv[1]);
  uint64_t sz_jk = parse_with_ss<uint64_t>(argv[2]);

  int n          = parse_with_ss<int>(argv[3]);

  vector<dbuffer_t> data;
  data.push_back(make_dbuffer(default_dtype(), sz_i * sz_jk));
  for(int i = 0; i != n; ++i) {
    data.push_back(make_dbuffer(default_dtype(), sz_jk*sz_jk));
    data.back().random("-0.1", "0.1");
  }

  uint64_t ni = sz_i;
  uint64_t nj = sz_jk;
  uint64_t nk = sz_jk;

  void* handle;
  auto status = mkl_jit_create_sgemm(
    &handle,
    MKL_ROW_MAJOR,
    MKL_NOTRANS, MKL_NOTRANS,
    ni, nk, nj,
    1.0, nj, nk, 0.0, nk);
  if(status == MKL_JIT_ERROR) {
    DOUT("JIT ERROR");
  } else if(status == MKL_NO_JIT) {
    DOUT("MKL NO JIT");
  } else if(status == MKL_JIT_SUCCESS) {
    DOUT("MKL JIT SUCCESS");
  } else {
    DOUT("??????????? UNKNOWN STATUS");
  }
  auto f = mkl_jit_get_sgemm_ptr(handle);

  {
    gremlin_t gremlin("mm test03 with jit");
    for(int i = 1; i != n+1; ++i) {
      dbuffer_t out = make_dbuffer(default_dtype(), sz_i*sz_jk);
      f(handle,
        reinterpret_cast<float*>(data[0].raw()),
        reinterpret_cast<float*>(data[i].raw()),
        reinterpret_cast<float*>(out.raw()));
      data[0] = out;
    }
  }

  // TODO: gotta free handle

  return 0;
}

int test04(int argc, char** argv) {
  if(argc != 7) {
    std::cout << "Usage: sz_i sz_jk n npart_i npart_jk nrunner" << std::endl;
    return 1;
  }

  uint64_t sz_i  = parse_with_ss<uint64_t>(argv[1]);
  uint64_t sz_jk = parse_with_ss<uint64_t>(argv[2]);

  int n          = parse_with_ss<int>(argv[3]);

  vector<dbuffer_t> data;
  data.push_back(make_dbuffer(default_dtype(), sz_i * sz_jk));
  for(int i = 0; i != n; ++i) {
    data.push_back(make_dbuffer(default_dtype(), sz_jk*sz_jk));
    data.back().random("-0.1", "0.1");
  }

  {
    gremlin_t gremlin("mm test04 with enable_instructions");
    for(int i = 1; i != n+1; ++i) {
      dbuffer_t out = make_dbuffer(default_dtype(), sz_i*sz_jk);
      matrix_multiply(
	default_dtype(),
	sz_i, sz_jk, sz_jk, false, false,
        reinterpret_cast<float*>(out.raw()),
        reinterpret_cast<float*>(data[0].raw()),
        reinterpret_cast<float*>(data[i].raw()));
      data[0] = out;
    }
  }

  return 0;
}
int main01(int argc, char** argv) {
  if(argc != 5) {
    std::cout << "Usage: sz n npart nrunner" << std::endl;
    return 1;
  }

  uint64_t sz = parse_with_ss<uint64_t>(argv[1]);
  int n       = parse_with_ss<int>(argv[2]);
  int npart   = parse_with_ss<int>(argv[3]);
  int nrunner = parse_with_ss<int>(argv[4]);

  auto [taskgraph, data] = make_tg_and_data(sz, n, npart);

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  execute_taskgraph_settings_t settings {
    .num_apply_runner = nrunner,
    .num_send_runner = 0,
    .num_recv_runner = 0
  };

  {
    std::cout << "starting execution..." << std::endl;
    gremlin_t _("execute sz=" + write_with_ss(sz) + ", n=" + write_with_ss(n));
    execute_taskgraph(taskgraph, settings, kernel_manager, nullptr, data);
  }

  return 0;
}

int main(int argc, char** argv) {
  test03(argc, argv);

  if(argc != 7) {
    std::cout << "Usage: sz_i sz_jk n npart_i npart_jk nrunner" << std::endl;
    return 1;
  }

  uint64_t sz_i  = parse_with_ss<uint64_t>(argv[1]);
  uint64_t sz_jk = parse_with_ss<uint64_t>(argv[2]);

  int n       = parse_with_ss<int>(argv[3]);

  int npart_i  = parse_with_ss<int>(argv[4]);
  int npart_jk = parse_with_ss<int>(argv[5]);

  int nrunner = parse_with_ss<int>(argv[6]);

  auto [taskgraph, data] = make_tg_and_data(sz_i, sz_jk, n, npart_i, npart_jk);

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  execute_taskgraph_settings_t settings {
    .num_apply_runner = nrunner,
    .num_send_runner = 0,
    .num_recv_runner = 0
  };

  {
    std::cout << "starting execution..." << std::endl;
    gremlin_t _("execute");
    execute_taskgraph(taskgraph, settings, kernel_manager, nullptr, data);
  }
}
