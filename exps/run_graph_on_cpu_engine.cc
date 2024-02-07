#include "../src/base/args.h"
#include "../src/base/copyregion.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/autoplace/apart.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/autoplace/alocate.h"
#include "../src/base/placement.h"
#include <fstream>
#include "../src/server/cpu/server.h"

using tensor_t     = graph_writer_t::tensor_t;

vector<placement_t> pre_assign_loc(vector<partition_t> const& parts, int num_locs, vector<int> placement, graph_t const& graph)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  int i = 0;
  for(auto const& part: parts) {
    ret.emplace_back(part);
    for(int& loc: ret.back().locations.get()) {
      loc = placement[i];
      i+=1;
    }
  }
  std::cout << "Total number of nodes: " << i << std::endl;
  return ret;
}

graph_t make_graph_ff_simple(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});

  tensor_t w1 = writer.input({hidden, dim});
  tensor_t w2 = writer.input({dim, hidden});
  tensor_t w3 = writer.input({hidden, dim});

  tensor_t w1t = w1.transpose(0, 1);
  tensor_t w2t = w2.transpose(0, 1);
  tensor_t w3t = w3.transpose(0, 1);

  scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

  tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

  tensor_t b = writer.matmul(x, w3t) ;

  tensor_t c = writer.mul(a, b);

  writer.matmul(c, w2t).save_inplace();

  return writer.get_graph();
}

vector<placement_t> create_same_loc_placements(vector<partition_t> const& parts, int num_locs)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  for(auto const& part: parts) {
    // std::cout <<  "part: " << part  <<  std::endl;
    // for(int i = 0; i <  part.num_parts(); i++){
    //   std::cout << part.total_shape()  <<  std::endl;
    // }
    
    ret.emplace_back(part);
  }
  return ret;
}

buffer_t make_out(vector<uint64_t> const& shape) {
  dbuffer_t dbuffer = make_dbuffer(dtype_t::f32, product(shape));
  return dbuffer.data;
}

buffer_t make_data(vector<uint64_t> const& shape) {
  buffer_t ret = make_out(shape);
  dbuffer_t(dtype_t::f32, ret).random("-0.00001", "0.00001");
  return ret;
}

void main_rank_zero(server_base_t* server, args_t& args, int world_size)
{
  // TODO:
  // 1. create a graph
  int num_threads_per = 2; // 
  DOUT("world_size*num_threads_per: " << world_size*num_threads_per);
  uint64_t batch = 1000;
  uint64_t hidden = 1000;
  uint64_t dim = 1000;
  graph_t const& graph = make_graph_ff_simple(batch,hidden,dim);

  std::ofstream f("my_graph.gv");
  graph.print_graphviz(f);
  vector<partition_t> parts = apart01(graph,world_size*num_threads_per);
  // vector<placement_t> placements = alocate01(graph, parts,world_size,1000);

  int num_devices;
  vector<int> rl_placements;
  // Open the input file
  std::ifstream inputFile("input.txt");

  if (!inputFile) {
      std::cerr << "Failed to open the file." << std::endl;
  }

  std::string line;
  // Read each line from the file
  std::getline(inputFile, line);
  std::istringstream iss(line);
  // Read the first integer from the line
  if (!(iss >> num_devices)) {
      std::cerr << "Failed to read the number of devices." << std::endl;
  }
  std::cout << "num_devices: " << num_devices << std::endl;

  while(std::getline(inputFile, line)){
    std::istringstream iss1(line);
    int num;
    // Read the remaining integers and store them in the vector
    while (iss1 >> num) {
      rl_placements.push_back(num);
    }
  }
  

  // // Open the file in write mode, which clears its contents
  // std::ofstream outputFile("input.txt", std::ios::out | std::ios::trunc);

  // if (!outputFile) {
  //     std::cerr << "Failed to open the file." << std::endl;
  // }

  // // Close the file
  // outputFile.close();

  std::cout << "Text file cleared successfully." << std::endl;
  
  // Print the values read
  std::cout << "Number of devices: " << num_devices << std::endl;
  std::cout << "Placements:" << std::endl;
  for (int placement : rl_placements) {
      std::cout << placement << " ";
  }
  std::cout << std::endl;
  
  std::vector<placement_t> placements = pre_assign_loc(parts, num_devices, rl_placements, graph);


  // 2. insert the input tensors
  buffer_t x = make_data({batch, dim});
  buffer_t w1 = make_data({hidden, dim});
  buffer_t w2 = make_data({dim, hidden});
  buffer_t w3 = make_data({hidden, dim});

  server->insert_tensor(0,placements[0],dbuffer_t(dtype_t::f32, x));
  server->insert_tensor(1,placements[1],dbuffer_t(dtype_t::f32, w1));
  server->insert_tensor(2,placements[2],dbuffer_t(dtype_t::f32, w2));
  server->insert_tensor(3,placements[3],dbuffer_t(dtype_t::f32, w3));

  // 3. execute the graph
  server->execute_graph(graph, placements);
}

int main(int argc, char** argv) {
  int expected_argc = 5;
  if(argc < expected_argc) {
    std::cout << "Need more arg.\n" << std::endl;
    return 1;
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  // int num_threads = std::max(1, int(std::thread::hardware_concurrency()));
  int num_threads = 12;

  int num_channels = 8;
  int num_channels_per_move = 1;

  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);

  int this_rank = communicator.get_this_rank();
  DOUT("this_rank: " << this_rank);

  args_t args(argc-(expected_argc-1), argv+(expected_argc-1));

  
  cpu_mg_server_t server(
    communicator, mem_size, num_threads, num_channels_per_move);

  if(is_rank_zero) {
    main_rank_zero(&server, args, world_size);
    server.shutdown();
  } else {
    server.listen();
  }
  return 0;
}