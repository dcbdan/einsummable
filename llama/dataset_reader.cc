#include "dataset_reader.h"

#include "../src/einsummable/touch.h"
#include "../src/engine/touch.h"

dataset_reader_t::dataset_reader_t(string tokenizer_filename, string data_filename)
  : piper("./llama_tokenizer", tokenizer_filename),
		file(data_filename, std::ios_base::in | std::ios_base::binary)
{
	vector<int> v = parse_vector<int>(piper.read());
	pad_id = v[0];
	vocab_size = v[1];

  while(file) {
    pos_type offset = file.tellg();
    file.read(reinterpret_cast<char*>(&sz), sizeof(int32_t));
    if(!file) {
      break;
    }
    if(sz <= 0) {
      throw std::runtime_error("sz < 0");
    }
    offsets.push_back(offset);
    file.seekg(sz, std::ios_base::cur);
  }

  file.clear();
}

vector<int> dataset_reader_t::read(int which) {
  if(which < 0 || which >= offsets.size()) {
    throw std::runtime_error("invalid which");
  }
  auto const& offset = offsets[which];
  file.seekg(offset, std::ios_base::beg);
  file.read(reinterpret_cast<char*>(&sz), sizeof(int32_t));
  string str(sz, ' ');
  file.read(str.data(), sz);
	{
		char const* raw = reinterpret_cast<char const*>(&sz);
		string msgsz(raw, raw + sizeof(sz));
	  piper.write(msgsz);
		piper.write(str);
	}

	vector<int> ret = parse_vector<int>(piper.read());
  if(ret.size() == 0) {
    throw std::runtime_error("empty number of tokens!");
  }
  return ret;
}

tuple<vector<int>, int>
dataset_reader_t::datum(int which, uint64_t seqlen)
{
  vector<int> tokens = read(which);

  if(tokens.size() >= seqlen + 1) {
    uint64_t start = 0;
    if(tokens.size() > seqlen + 1) {
      start = runif(tokens.size() - (seqlen + 1));
    }
    auto beg = tokens.begin() + start;
    vector<int> ts(beg, beg + seqlen);
    int label = tokens[start + seqlen];
    return {ts, label};
  } else {
    vector<int> ts(seqlen);
    uint64_t start = (seqlen + 1) - tokens.size();
    for(uint64_t i = 0; i != start; ++i) {
      ts[i] = pad_id;
    }
    for(uint64_t i = 0; i != tokens.size() - 1; ++i) {
      ts[start + i] = tokens[i];
    }
    int label = tokens.back();
    return {ts, label};
  }
}

tuple<vector<int>, int>
dataset_reader_t::random_datum(uint64_t seqlen)
{
  return datum(runif(num()), seqlen);
}

tuple<vector<vector<int>>, vector<int>>
dataset_reader_t::random_data(uint64_t batch_size, uint64_t seqlen)
{
  vector<vector<int>> xs;
  vector<int> ys;
  xs.reserve(batch_size);
  ys.reserve(batch_size);
  for(uint64_t i = 0; i != batch_size; ++i) {
    auto [x,y] = random_datum(seqlen);
    xs.push_back(x);
    ys.push_back(y);
  }
  return {xs,ys};
}

dbuffer_t
dataset_reader_t::make_embedding(
  dbuffer_t const& embedding_matrix,
  vector<int> const& tokens)
{
  if(tokens.size() == 0) {
    throw std::runtime_error("make_embedding: cannot have empty tokens");
  }
  uint64_t seqlen = tokens.size();

  if(embedding_matrix.nelem() % vocab_size != 0) {
    throw std::runtime_error("invalid size embedding matrix");
  }
  uint64_t embed_size = embedding_matrix.nelem() / vocab_size;

  touchdim_t td_embed {
    .d_inn = embed_size,
    .d_out = embed_size,
    .offset_inn = 0,
    .offset_out = 0,
    .size = embed_size
  };
  auto make_select_td = [&](int token, int which) {
    return touchdim_t {
      .d_inn = vocab_size,
      .d_out = seqlen,
      .offset_inn = uint64_t(token),
      .offset_out = uint64_t(which),
      .size = 1
    };
  };

  dbuffer_t ret = make_dbuffer(embedding_matrix.dtype, seqlen * embed_size);
  ret.zeros();

  for(uint64_t i = 0; i != seqlen; ++i) {
    auto const& token = tokens[i];
    if(token < 0) {
      // for negative tokens (i.e. if pad_id = -1), then just set to all zeros
      continue;
    }
    touch_t touch {
      .selection = {make_select_td(token, i), td_embed},
      .castable = std::nullopt,
      .dtype = embedding_matrix.dtype
    };

    execute_touch(touch, ret.raw(), embedding_matrix.raw());
  }

  return ret;
}

dbuffer_t dataset_reader_t::one_hot_encode(
  dtype_t dtype,
  vector<int> const& tokens)
{
  uint64_t seqlen = tokens.size();
  dbuffer_t ret = make_dbuffer(dtype, seqlen*vocab_size);
  ret.zeros();
  scalar_t one = scalar_t::one(dtype);
  for(uint64_t s = 0; s != seqlen; ++s) {
    uint64_t token = tokens[s];
    if(token >= 0) {
      if(token >= vocab_size) {
        throw std::runtime_error("invalid token value");
      }
      ret.set(s * vocab_size + token, one);
    }
  }
  return ret;
}
