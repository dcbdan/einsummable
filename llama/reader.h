#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/dbuffer.h"
#include "../src/einsummable/relation.h"

#include "../src/execution/cpu/mpi_class.h"

#include <fstream>

struct local_tensor_reader_t {
  local_tensor_reader_t(string const& filename);

  vector<string> all_names();

  buffer_t operator()(string const& tensor_name);

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
    mpi_t* mpi, map<int, buffer_t>& d,
    string const& base_filename, int n_total_files);

  // Should only be called by rank zero {{{
  relation_t operator()(
    string const& tensor_name,
    vector<uint64_t> const& shape,
    int starting_tid);

  void shutdown();
  // }}}

  void listen();

  int const& num_files() { return n_total_files; }
private:
  void _shutdown();

  void _read(string const& tensor_name, vector<int> const& whiches);

  placement_t get_placement(string const& str, vector<uint64_t> const& shape) const;

  int world_size;
  int n_total_files;

  mpi_t* mpi;
  map<int, buffer_t>& data;

  map<int, local_tensor_reader_t> readers;
};

