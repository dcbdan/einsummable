#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/trainer.h"

#include "../src/autoplace/autoplace.h"

void usage() {
  std::cout << "Usage: addr_zero is_client world_size memsize (Args)\n"
               "Args:\n";
}

void main_rank_zero(
  std::unique_ptr<server_base_t>& server,
  args_t& args)
{
  string tokenizer_file = args.get<string>("tokenizer");
  string dataset_file   = args.get<string>("dataset");

  dataset_reader_t reader(tokenizer_file, dataset_file);

  auto [x,y] = reader.make_random_data(dtype_t::f32, 1, 10000);
  DLINEOUT("AAAAA");
}

int main(int argc, char** argv) {
  if(argc < 4) {
    usage();
    return 1;
  }
  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  int num_threads = 8;//2; //std::max(1, int(std::thread::hardware_concurrency()));
  int num_channels = 4;
  int num_channels_per_move = 1;

  DOUT("n_locs " << world_size << " | num_threads_per_loc " << num_threads);

  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  std::unique_ptr<server_base_t> server;
  server = std::unique_ptr<server_base_t>(
    new cpu_tg_server_t(communicator, mem_size, num_threads));

  if(!is_rank_zero) {
    server->listen();
    return 0;
  }

  args_t args(argc-4, argv+4);

  main_rank_zero(server, args);

  server->shutdown();
}

