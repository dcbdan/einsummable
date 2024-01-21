#include "dataset_reader.h"

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

tuple<dbuffer_t, dbuffer_t>
dataset_reader_t::make_random_data(
  dtype_t dtype,
  uint64_t bsz,
  uint64_t max_seq_len)
{
  dbuffer_t x = make_dbuffer(dtype, bsz * max_seq_len * vocab_size);
  x.zeros();

  dbuffer_t y = make_dbuffer(dtype, bsz *               vocab_size);
  y.zeros();

  scalar_t one = scalar_t::one(dtype);

  auto set_x = [&](uint64_t b, uint64_t s, uint64_t v) {
    uint64_t iii = indexer_utils<uint64_t>::idxs_to_index(
      {bsz, max_seq_len, vocab_size},
      {b,s,v});
    x.set(iii, one);
  };
  auto set_y = [&](uint64_t b, uint64_t v) {
    uint64_t iii = indexer_utils<uint64_t>::idxs_to_index(
      {bsz, vocab_size},
      {b,v});
    y.set(iii, one);
  };

  for(int b = 0; b != bsz; ++b) {
    vector<int> tokens = random();
    if(tokens.size() >= max_seq_len + 1) {
      uint64_t start = 0;
      if(tokens.size() > max_seq_len + 1) {
        start = runif(tokens.size() - (max_seq_len + 1));
      }
      for(uint64_t i = 0; i != max_seq_len; ++i) {
        set_x(b, i, tokens[start + i]);
      }
      set_y(b, tokens[start + max_seq_len]);
    } else {
      uint64_t start = (max_seq_len + 1) - tokens.size();
      if(pad_id >= 0) {
        for(uint64_t i = 0; i != start; ++i) {
          set_x(b, i, pad_id);
        }
      }
      for(uint64_t i = 0; i != tokens.size() - 1; ++i) {
        set_x(b, start + i, tokens[i]);
      }
      set_y(b, tokens.back());
    }
  }

  return {x,y};
}

