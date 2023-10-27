#include "../src/engine/communicator.h"

struct silly_msg_t {
  int src;
  bool done;
};

int main(int argc, char** argv) {
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
