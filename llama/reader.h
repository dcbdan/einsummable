#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/dbuffer.h"
#include "../src/einsummable/relation.h"

#include "../src/engine/communicator.h"
#include "../src/server/cpu/server.h"

#include <fstream>

struct local_tensor_reader_t {
  local_tensor_reader_t(string const& filename);

  vector<string> all_names();

  buffer_t read(string const& tensor_name);
  buffer_t operator()(string const& tensor_name, dtype_t dtype);

private:
  std::ifstream file;

  void to_beginning();

  optional<tuple<string, uint64_t>>
  read_next_weight_info();
};

struct tensor_reader_t {
  // this reads files base_filename_ii
  // where ii is 2-digit str of n (n=1 -> 01, n=19 -> 19)
  tensor_reader_t(
    communicator_t& comm,
    std::function<void(map<int, buffer_t> const&)> process,
    int this_rank, int world_size,
    string const& base_filename, int n_total_files,
    dtype_t dtype = default_dtype());

  static placement_t get_placement(
    string const& tensor_name,
    vector<uint64_t> const& shape,
    int world_size,
    int n_total_files);

  // Should only be called by rank zero {{{
  relation_t operator()(
    string register_cmd,
    string const& tensor_name,
    vector<uint64_t> const& shape,
    int starting_tid);

  void shutdown(string register_cmd);
  // }}}

  void listen_read();

  void listen_shutdown();

  int const& num_files() { return n_total_files; }

  static string read_cmd()     { return "tensor_reader_t/read";     }
  static string shutdown_cmd() { return "tensor_reader_t/shutdown"; }
private:
  void _shutdown();

  map<int, buffer_t> _read(
    string const& tensor_name, vector<int> const& whiches);

  communicator_t& comm;
  std::function<void(map<int, buffer_t> const&)> process_data;

  int world_size;
  int n_total_files;

  map<int, local_tensor_reader_t> readers;

  dtype_t dtype;
};

