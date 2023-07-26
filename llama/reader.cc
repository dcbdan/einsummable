#include "reader.h"

local_tensor_reader_t::local_tensor_reader_t(string const& filename) :
  file(filename, std::ios::binary)
{
  if(!file) {
    throw std::runtime_error("Failed to open the file.");
  }
}

vector<string> local_tensor_reader_t::all_names() {
  vector<string> ret;
  while(true) {
    auto maybe_info = read_next_weight_info();
    if(!maybe_info) {
      break;
    }
    auto const& [name, nelem] = maybe_info.value();
    ret.push_back(name);

    // Assuming all tensors are stored in float16s
    uint64_t size = nelem * sizeof(float16_t);

    file.seekg(file.tellg() + std::ifstream::pos_type(size));
    // TODO: how to use pos_type correctly?
  }
  to_beginning();
  return ret;
}

buffer_t local_tensor_reader_t::operator()(string const& tensor_name) {
  while(true) {
    auto maybe_info = read_next_weight_info();
    if(!maybe_info) {
      break;
    }

    auto const& [name, nelem] = maybe_info.value();

    // Assuming all tensors are stored in float16s
    uint64_t size = nelem * sizeof(float16_t);

    if(name == tensor_name) {
      // Read the tensor data
      buffer_t buffer = make_buffer(size);
      file.read(reinterpret_cast<char*>(buffer->data), size);

      to_beginning();
      return buffer;
    } else {
      file.seekg(file.tellg() + std::ifstream::pos_type(size));
      // TODO: how to use pos_type correctly?
    }
  }

  to_beginning();
  throw std::runtime_error("did not find \"" + tensor_name + "\"");
}

void local_tensor_reader_t::to_beginning() {
  file.clear();
  file.seekg(0);
}

optional<tuple<string, uint64_t>>
local_tensor_reader_t::read_next_weight_info() {
  if(file.eof()) {
    return std::nullopt;
  }

  // Read the text data (name of weight tensor)
  char text_data[51];
  file.read(text_data, 50);
  text_data[50] = '\0';
  std::string name(text_data);

  if(file.eof()) {
    return std::nullopt;
  }

  std::string space = " ";
  const auto str_end = name.find_last_not_of(space);
  const auto str_range = str_end + 1;
  name = name.substr(0, str_range);

  // Read the binary data (size of tensor)
  int64_t nelem;
  file.read(reinterpret_cast<char*>(&nelem), sizeof(int64_t));
  return optional<tuple<string, uint64_t>>({name, nelem});
}

tensor_reader_t::tensor_reader_t(
  int this_rank, int world_size,
  string const& base_filename, int n)
  : world_size(world_size), n_total_files(n)
{
  for(int i = this_rank; i < n; i += world_size) {
    string si = write_with_ss(i);
    if(i < 10) {
      si = "0" + si;
    }
    string filename = base_filename + "_" + si;

    readers.insert({
      i, local_tensor_reader_t(filename)
    });
  }
}

relation_t tensor_reader_t::operator()(
  string register_cmd, mpi_t* mpi, map<int, buffer_t>& data,
  string const& name,
  vector<uint64_t> const& shape,
  int starting_tid)
{
  if(mpi && mpi->this_rank != 0) {
    throw std::runtime_error("only rank 0 should be getting the relation");
  }

  // if this is a norm, only load it here on rank 0
  if(name.find("norm") != string::npos) {
    auto& reader = readers.at(0);

    placement_t pl(partition_t::singleton(shape));

    data.insert({starting_tid, reader(name)});

    vector<int> block_shape(shape.size(), 1);
    vtensor_t<int> tids(block_shape, {starting_tid});

    return relation_t {
      .dtype     = dtype_t::f16,
      .placement = pl,
      .tids      = tids
    };
  }

  // Everything that isn't a norm is partitioned into column strips
  // or into row strips.
  auto placement = get_placement(name, shape);
  auto block_shape = placement.block_shape();

  // construct the tids
  vtensor_t<int> tids(block_shape);
  auto& v_tids = tids.get();
  std::iota(v_tids.begin(), v_tids.end(), starting_tid);

  // send the tensor name & tids
  for(int dst = 1; dst != world_size; ++dst) {
    mpi->send_str(register_cmd, dst);
    mpi->send_str(read_cmd(), dst);
    mpi->send_str(name, dst);
    mpi->send_vector(v_tids, dst);
  }

  _read(name, v_tids, data);

  return relation_t {
    .dtype     = dtype_t::f16,
    .placement = placement,
    .tids      = tids
  };
}

void tensor_reader_t::shutdown(string reg_cmd, mpi_t* mpi) {
  if(!mpi) { return; }

  if(mpi->this_rank != 0) {
    throw std::runtime_error("only rank 0 should do broadcasting");
  }

  for(int i = 1; i != mpi->world_size; ++i) {
    mpi->send_str(reg_cmd, i);
    mpi->send_str(shutdown_cmd(), i);
  }

  _shutdown();
}

void tensor_reader_t::listen_read(mpi_t* mpi, map<int, buffer_t>& data) {
  if(!mpi) {
    throw std::runtime_error("should not call listen if mpi is not setup");
  }
  if(mpi->this_rank == 0) {
    throw std::runtime_error("rank zero should not call listen method");
  }

  string tensor_name = mpi->recv_str(0);
  vector<int> file_to_tid = mpi->recv_vector<int>(0);
  _read(tensor_name, file_to_tid, data);
}

void tensor_reader_t::listen_shutdown() {
  _shutdown();
}

void tensor_reader_t::_shutdown() {
  readers.clear();
}

void tensor_reader_t::_read(
  string const& tensor_name, vector<int> const& whiches,
  map<int, buffer_t>& data)
{
  for(auto& [i, reader]: readers) {
    data.insert_or_assign(whiches[i], reader(tensor_name));
  }
}

placement_t tensor_reader_t::get_placement(
  string const& name,
  vector<uint64_t> const& shape) const
{
  vector<int> locs;
  locs.reserve(n_total_files);
  for(int i = 0; i != n_total_files; ++i) {
    locs.push_back(i % world_size);
  }

  partition_t partition = partition_t::singleton(shape);
  auto fix = [&](int d) {
    partition.partdims[d] = partdim_t::repeat(
      n_total_files,
      uint64_div(shape[d], n_total_files, name));
  };

  auto name_has = [&](vector<string> xs) {
    for(auto const& x: xs) {
      if(name.find(x) != std::string::npos) {
        return true;
      }
    }
    return false;
  };

  if(name == "output.weight") {
    // concat matrix dim: 0
    // shape: (vocab size, (n_heads, head_dim))
    fix(0);
  } else if(name == "tok_embeddings.weight") {
    // concat matrix dim: 1
    // shape: (vocab size, dim)
    fix(1);
  } else if(name_has({"wq", "wk", "wv"})) {
    // concat matrix dim: 0
    // ( (n_heads, head_dim), (n_heads, head_dim) )
    fix(0);
  } else if(name_has({"wo"})) {
    // concat matrix dim: 1
    // ( (n_heads, head_dim), (n_heads, head_dim) )
    fix(2);
  } else if(name_has({"w1", "w3"})) {
    // concat matrix dim: 0
    // ( hidden, (n_heads, head_dim) )
    fix(0);
  } else if(name_has({"w2"})) {
    // concat matrix dim: 1
    // ( (n_heads, head_dim), hidden )
    fix(2);
  } else {
    throw std::runtime_error("couldn't handle name \"" + name + "\"");
  }

  return placement_t(
    partition,
    vtensor_t<int>(partition.block_shape(), locs));
}

