#include "reader.h"

local_tensor_reader_t::local_tensor_reader_t(string const& filename) :
  file(filename, std::ios::binary)
{
  if(!file) {
    throw std::runtime_error("Failed to open the file:" + filename);
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

buffer_t local_tensor_reader_t::operator()(
  string const& tensor_name,
  dtype_t expected_dtype)
{
  buffer_t ret = read(tensor_name);
  if(expected_dtype != dtype_t::f16) {
    return dbuffer_t(dtype_t::f16, ret).copy(expected_dtype).data;
  } else {
    return ret;
  }
}

buffer_t local_tensor_reader_t::read(string const& tensor_name) {
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
  communicator_t& comm,
  std::function<void(map<int, buffer_t> const&)> process,
  int this_rank, int world_size,
  string const& base_filename, int n,
  dtype_t d)
  : comm(comm), process_data(process), world_size(world_size), n_total_files(n),
    dtype(d)
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
  string register_cmd,
  string const& name,
  vector<uint64_t> const& shape,
  int starting_tid)
{
  if(comm.get_this_rank() != 0) {
    throw std::runtime_error("only rank 0 should be getting the relation");
  }

  // if this is a norm, only load it here on rank 0
  if(name.find("norm") != string::npos) {
    auto& reader = readers.at(0);

    placement_t pl(partition_t::singleton(shape));

    process_data(map<int, buffer_t>{ {starting_tid, reader(name, dtype) } });

    vector<int> block_shape(shape.size(), 1);
    vtensor_t<int> tids(block_shape, {starting_tid});

    return relation_t {
      .dtype     = dtype,
      .placement = pl,
      .tids      = tids
    };
  }

  // Everything that isn't a norm is partitioned into column strips
  // or into row strips.
  auto placement = get_placement(name, shape, world_size, n_total_files);
  auto block_shape = placement.block_shape();

  // construct the tids
  vtensor_t<int> tids(block_shape);
  auto& v_tids = tids.get();
  std::iota(v_tids.begin(), v_tids.end(), starting_tid);

  // send the tensor name & tids
  for(int dst = 1; dst != world_size; ++dst) {
    comm.send_string(dst, register_cmd);
    comm.send_string(dst, read_cmd());
    comm.send_string(dst, name);
    comm.send_vector(dst, v_tids);
  }

  process_data(_read(name, v_tids));

  return relation_t {
    .dtype     = dtype,
    .placement = placement,
    .tids      = tids
  };
}

void tensor_reader_t::shutdown(string reg_cmd) {
  if(comm.get_this_rank() != 0) {
    throw std::runtime_error("only rank 0 should do broadcasting");
  }

  for(int dst = 1; dst != world_size; ++dst) {
    comm.send_string(dst, reg_cmd);
    comm.send_string(dst, shutdown_cmd());
  }

  _shutdown();
}

void tensor_reader_t::listen_read() {
  if(comm.get_this_rank() == 0) {
    throw std::runtime_error("rank zero should not call listen method");
  }

  string tensor_name = comm.recv_string(0);
  vector<int> file_to_tid = comm.recv_vector<int>(0);
  process_data(_read(tensor_name, file_to_tid));
}

void tensor_reader_t::listen_shutdown() {
  _shutdown();
}

void tensor_reader_t::_shutdown() {
  readers.clear();
}

map<int, buffer_t> tensor_reader_t::_read(
  string const& tensor_name,
  vector<int> const& whiches)
{
  map<int, buffer_t> data;
  for(auto& [i, reader]: readers) {
    data.insert_or_assign(whiches[i], reader(tensor_name, dtype));
  }
  return data;
}

placement_t tensor_reader_t::get_placement(
  string const& name,
  vector<uint64_t> const& shape,
  int world_size,
  int n_total_files)
{
  if(name.find("norm") != string::npos) {
    return placement_t(partition_t::singleton(shape));
  }
  /////////

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

/////////////////////////////////////////////////////////////////////////////////

tensor_reader2_t::tensor_reader2_t(
  int num_buffers,
  string const& base_filename, 
  int n_total_files,
  dtype_t dtype)
  : num_buffers(num_buffers), n_total_files(n_total_files), dtype(dtype)
{
  for(int i = 0; i != n_total_files; ++i) {
    string si = write_with_ss(i);
    if(i < 10) {
      si = "0" + si;
    }
    string filename = base_filename + "_" + si;

    readers.emplace_back(filename);
  }
}

placement_t tensor_reader2_t::get_placement(
  string const& tensor_name,
  vector<uint64_t> const& shape,
  int n_total_files)
{
  return tensor_reader_t::get_placement(tensor_name, shape, num_buffers, n_total_files);
}

tuple<relation_t, vector<buffer_t>> tensor_reader2_t::operator()(
  string const& name,
  vector<uint64_t> const& shape,
  int starting_tid)
{
  // if this is a norm, only load it here on rank 0
  if(name.find("norm") != string::npos) {
    vector<buffer_t> data;
    data.push_back(readers.at(0)(name, dtype));

    placement_t pl(partition_t::singleton(shape));

    vector<int> block_shape(shape.size(), 1);
    vtensor_t<int> tids(block_shape, {starting_tid});

    relation_t rel {
      .dtype     = dtype,
      .placement = pl,
      .tids      = tids
    };
    return {rel, data};
  }

  // Everything that isn't a norm is partitioned into column strips
  // or into row strips.
  auto placement = get_placement(name, shape, n_total_files);
  auto block_shape = placement.block_shape();

  // construct the tids
  vtensor_t<int> tids(block_shape);
  auto& v_tids = tids.get();
  std::iota(v_tids.begin(), v_tids.end(), starting_tid);

  vector<buffer_t> data;
  for(auto& reader: readers) {
    data.push_back(reader(name, dtype));
  }

  // How it works is there is one reader for each file,
  // and each file contains a portion of our tensor,
  // and the placement should be split too.

  relation_t rel {
    .dtype     = dtype,
    .placement = placement,
    .tids      = tids
  };
  return {rel, data};
}

