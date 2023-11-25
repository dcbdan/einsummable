#include "../src/engine/communicator.h"

struct silly_msg_t {
  int src;
  bool done;
};

void main01(int argc, char** argv) {
  if(argc < 4) {
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  communicator_t comm(addr_zero, is_rank_zero, world_size, 1);

  int this_rank = comm.get_this_rank();
  int channel = 0;

  if(this_rank == 0 && world_size >= 2) {
    int mine = 99751;
    int peer = -1;

    comm.send_int(1, channel, mine);
    peer = comm.recv_int(1, channel);

    DOUT("this is rank 0, peer number is " << peer);
  } else if(this_rank == 1 && world_size >= 2) {
    int mine = 99882;
    int peer = -1;

    peer = comm.recv_int(0, channel);
    comm.send_int(0, channel, mine);

    DOUT("this is rank 1, peer number is " << peer);
  }

  comm.barrier();

  bool constant_poll = false;

  auto start = clock_now();

  map<int, int> counts;
  comm.start_listen_notify_type<silly_msg_t>(
    [&counts](silly_msg_t const& msg) -> bool
    {
      auto const& [src, done] = msg;
      counts[src]++;
      return done;
    },
    constant_poll
  );

  for(int i = 0; i != 100000; ++i) {
    int dst = runif(world_size-1);
    if(dst >= this_rank) {
      dst += 1;
    }
    comm.notify(dst, silly_msg_t { this_rank, false });
  }

  for(int rank = 0; rank != world_size; ++rank) {
    if(rank != this_rank) {
      comm.notify(rank, silly_msg_t { this_rank, true });
    }
  }

  comm.stop_listen_notify();
  auto end = clock_now();

  for(auto const& [src, count]: counts) {
    DOUT("From " << src << ": " << count << " number of messages");
  }
  double total_time = std::chrono::duration<double>(end-start).count();
  DOUT("Total time: " << total_time);
}

// send form 0 to 1
void main02(int argc, char** argv) {
  if(argc < 4) {
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  if(world_size != 2) {
    throw std::runtime_error("expect world size == 2");
  }

  communicator_t comm(addr_zero, is_rank_zero, world_size, 1);

  int this_rank = comm.get_this_rank();

  uint64_t GB = 1000000000;
  uint64_t nGB = 3;

  vector<uint8_t> data(nGB*GB, this_rank == 0 ? 9 : 8);
  auto start = clock_now();
  if(this_rank == 0) {
    DOUT(int(data[0]) << " " << int(data.back()));
    comm.send(1, data.data(), data.size());
  } else if(this_rank == 1) {
    comm.recv(0, data.data(), data.size());
    DOUT(int(data[0]) << " " << int(data.back()));
  }
  comm.barrier();
  auto end = clock_now();

  double total_time = std::chrono::duration<double>(end-start).count();
  double bandwidth = double(nGB) / total_time;
  DOUT("Bandwidth: " << bandwidth << " GB/s");
}

// send from 0 to 1 then 1 to 0
void main03(int argc, char** argv) {
  if(argc < 4) {
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  if(world_size != 2) {
    throw std::runtime_error("expect world size == 2");
  }

  int nchannels = 1;
  communicator_t comm(addr_zero, is_rank_zero, world_size, nchannels);

  int this_rank = comm.get_this_rank();

  uint64_t GB = 1000000000;
  uint64_t nGB = 3;

  vector<uint8_t> d0(nGB*GB, this_rank == 0 ? 9 : 8);
  vector<uint8_t> d1(nGB*GB, this_rank == 0 ? 9 : 8);
  auto start = clock_now();
  if(this_rank == 0) {
    comm.send(1, d0.data(), d0.size());
    comm.recv(1, d1.data(), d1.size());
  } else if(this_rank == 1) {
    comm.recv(0, d0.data(), d0.size());
    comm.send(0, d1.data(), d1.size());
  }
  comm.barrier();
  auto end = clock_now();

  double total_time = std::chrono::duration<double>(end-start).count();
  double bandwidth = double(2*nGB) / total_time;
  DOUT("Bandwidth: " << bandwidth << " GB/s");
}

// send from 0 to 1 _and_ 1 to 0
void main04(int argc, char** argv) {
  if(argc < 4) {
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  if(world_size != 6) {
    throw std::runtime_error("expect world size == 6");
  }

  int nchannels = 8;
  communicator_t comm(addr_zero, is_rank_zero, world_size, nchannels);

  int this_rank = comm.get_this_rank();

  uint64_t GB = 1000000000;
  uint64_t nGB = 1;

  vector<vector<uint8_t>> ds0;
  vector<vector<uint8_t>> ds1;
  ds0.reserve(nchannels);
  ds1.reserve(nchannels);
  for(int i = 0; i != nchannels; ++i) {
    ds0.emplace_back(nGB*GB, this_rank == 0 ? 9 : 8);
    ds1.emplace_back(nGB*GB, this_rank == 0 ? 9 : 8);
  }

  auto start = clock_now();
  if(this_rank == 0) {
    vector<std::future<void>> fs;
    fs.reserve(nchannels);
    for(int i = 0; i != nchannels; ++i) {
      fs.push_back(comm.send_async(1, i, ds0[i].data(), ds0[i].size()));
      fs.push_back(comm.recv_async(1, i, ds1[i].data(), ds1[i].size()));
    }
    for(auto& f: fs) {
      f.get();
    }
  } else if(this_rank == 1) {
    vector<std::future<void>> fs;
    fs.reserve(nchannels);
    for(int i = 0; i != nchannels; ++i) {
      fs.push_back(comm.send_async(0, i, ds0[i].data(), ds0[i].size()));
      fs.push_back(comm.recv_async(0, i, ds1[i].data(), ds1[i].size()));
    }
    for(auto& f: fs) {
      f.get();
    }
  } else if(this_rank == 2) {
    vector<std::future<void>> fs;
    fs.reserve(nchannels);
    for(int i = 0; i != nchannels; ++i) {
      fs.push_back(comm.send_async(3, i, ds0[i].data(), ds0[i].size()));
      fs.push_back(comm.recv_async(3, i, ds1[i].data(), ds1[i].size()));
    }
    for(auto& f: fs) {
      f.get();
    }
  } else if(this_rank == 3) {
    vector<std::future<void>> fs;
    fs.reserve(nchannels);
    for(int i = 0; i != nchannels; ++i) {
      fs.push_back(comm.send_async(2, i, ds0[i].data(), ds0[i].size()));
      fs.push_back(comm.recv_async(2, i, ds1[i].data(), ds1[i].size()));
    }
    for(auto& f: fs) {
      f.get();
    }
  } else if(this_rank == 4) {
    vector<std::future<void>> fs;
    fs.reserve(nchannels);
    for(int i = 0; i != nchannels; ++i) {
      fs.push_back(comm.send_async(5, i, ds0[i].data(), ds0[i].size()));
      fs.push_back(comm.recv_async(5, i, ds1[i].data(), ds1[i].size()));
    }
    for(auto& f: fs) {
      f.get();
    }
  } else if(this_rank == 5) {
    vector<std::future<void>> fs;
    fs.reserve(nchannels);
    for(int i = 0; i != nchannels; ++i) {
      fs.push_back(comm.send_async(4, i, ds0[i].data(), ds0[i].size()));
      fs.push_back(comm.recv_async(4, i, ds1[i].data(), ds1[i].size()));
    }
    for(auto& f: fs) {
      f.get();
    }
  }

  comm.barrier();
  auto end = clock_now();

  double total_time = std::chrono::duration<double>(end-start).count();
  double bandwidth = double(2*nchannels*nGB) / total_time;
  DOUT("Bandwidth: " << bandwidth << " GB/s | this_rank = " << this_rank);
}

void main05(int argc, char** argv) {
  if(argc < 4) {
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  if(world_size < 2) {
    throw std::runtime_error("expect world size >= 2");
  }

  int nchannels = 4;
  communicator_t comm(addr_zero, is_rank_zero, world_size, nchannels);

  int this_rank = comm.get_this_rank();

  uint64_t GB = 1000000000;
  uint64_t nGB = 1;

  vector<vector<uint8_t>> ds0;
  vector<vector<uint8_t>> ds1;
  ds0.reserve(nchannels);
  ds1.reserve(nchannels);
  for(int i = 0; i != nchannels; ++i) {
    ds0.emplace_back(nGB*GB, this_rank == 0 ? 9 : 8);
    ds1.emplace_back(nGB*GB, this_rank == 0 ? 9 : 8);
  }

  comm.barrier();
  for(int i = 0;   i != world_size-1; ++i) {
  for(int j = i+1; j != world_size;   ++j) {
    if(this_rank == i) {
      auto start = clock_now();
      vector<std::future<void>> fs;
      fs.reserve(nchannels);
      for(int c = 0; c != nchannels; ++c) {
        fs.push_back(comm.send_async(j, c, ds0[c].data(), ds0[c].size()));
        fs.push_back(comm.recv_async(j, c, ds1[c].data(), ds1[c].size()));
      }
      for(auto& f: fs) {
        f.get();
      }
      auto end = clock_now();
      double total_time = std::chrono::duration<double>(end-start).count();
      double bandwidth = double(nchannels*2*nGB) / total_time;
      DOUT("send bw (" << i << "->" << j << "): " << bandwidth << " GB/s.");
    } else if(this_rank == j) {
      auto start = clock_now();
      vector<std::future<void>> fs;
      fs.reserve(nchannels);
      for(int c = 0; c != nchannels; ++c) {
        fs.push_back(comm.send_async(i, c, ds0[c].data(), ds0[c].size()));
        fs.push_back(comm.recv_async(i, c, ds1[c].data(), ds1[c].size()));
      }
      for(auto& f: fs) {
        f.get();
      }
      auto end = clock_now();
      double total_time = std::chrono::duration<double>(end-start).count();
      double bandwidth = double(nchannels*2*nGB) / total_time;
      DOUT("recv bw (" << i << "->" << j << "): " << bandwidth << " GB/s.");
    }

    comm.barrier();
  }}
}

void main06_send(communicator_t& comm, int channel, int num_send, uint64_t msg_size) {
  vector<uint64_t> data(msg_size);

  for(int i = 0; i != num_send; ++i) {
    comm.send(1, channel, data.data(), data.size());
    DLINEOUT(i + 1 << " @ channel " << channel);
  }
}

void main06_recv(communicator_t& comm, int channel, int num_recv, uint64_t msg_size) {
  vector<uint64_t> data(msg_size);

  for(int i = 0; i != num_recv; ++i) {
    comm.recv(0, channel, data.data(), data.size());
  }
}

void main06(int argc, char** argv) {
  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  int nchannels = parse_with_ss<int>(argv[4]);

  communicator_t comm(addr_zero, is_rank_zero, world_size, nchannels);

  uint64_t msg_size = 1000000;
  int num_msg = 100;
  int this_rank = comm.get_this_rank();

  vector<std::thread> threads;
  if(this_rank == 0) {
    for(int i = 0; i != nchannels; ++i) {
      threads.emplace_back([&, i] { main06_send(comm, i, num_msg, msg_size); });
    }
  } else if(this_rank == 1) {
    for(int i = 0; i != nchannels; ++i) {
      threads.emplace_back([&, i] { main06_recv(comm, i, num_msg, msg_size); });
    }
  }

  for(auto& t: threads) {
    t.join();
  }
}

#include "../src/engine/repartition.h"
#include "../src/einsummable/dbuffer.h"
relation_t make_dst_rel(
  int num_split, uint64_t nrow, uint64_t ncol)
{
  partition_t partition(vector<partdim_t>{
    partdim_t::split(nrow, num_split),
    partdim_t::split(ncol, 1)
  });
  placement_t placement(partition);
  {
    auto& locs = placement.locations.get();
    std::fill(locs.begin(), locs.end(), 1);
  }

  vtensor_t<int> tids(partition.block_shape());
  std::iota(tids.get().begin(), tids.get().end(), 0);

  return relation_t {
    .dtype = dtype_t::f32,
    .placement = placement,
    .tids = tids
  };
}

void main07(int argc, char** argv) {
  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  int nchannels = parse_with_ss<int>(argv[4]);

  communicator_t comm(addr_zero, is_rank_zero, world_size, nchannels);

  int this_rank = comm.get_this_rank();

  int nsplit = 100;
  uint64_t nrow = 10000;
  uint64_t ncol = 10000;

  relation_t dst_relation = make_dst_rel(nsplit, nrow, ncol);

  remap_relations_t remap;
  remap.insert(
    dst_relation.as_singleton(99),
    dst_relation);

  for(int i = 0; i != 100; ++i) {
    DOUT("starting " << (i+1) << " / 100");
    if(this_rank == 0) {
      dbuffer_t tensor = make_dbuffer(dtype_t::f32, nrow*ncol);
      tensor.ones();
      map<int, buffer_t> data;
      data.insert({99, tensor.data});

      repartition(comm, remap, data);
    } else if(this_rank == 1) {
      map<int, buffer_t> data;
      repartition(comm, remap, data);
    }

    comm.barrier();
  }

  DLINEOUT("done @ rank " << this_rank);
}

int main(int argc, char** argv) {
  main07(argc, argv);
}

