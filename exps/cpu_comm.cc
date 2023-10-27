#include "../src/engine/communicator.h"

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
}
