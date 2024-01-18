#include <string>
#include <iostream>

#include "../src/base/setup.h"
#include "../src/base/args.h"
#include "../src/base/indexer.h"
#include "../src/base/timetracker.h"

#include "../src/einsummable/dbuffer.h"

#include "sentencepiece_processor.h"

#include <fstream>

struct dataset_reader_t {
  dataset_reader_t(string tokenizer_filename, string data_filename)
    : file(data_filename, std::ios_base::in | std::ios_base::binary)
  {
    const auto status = processor.Load(tokenizer_filename);
    if(!status.ok()) {
      throw std::runtime_error("could not open " + write_with_ss(tokenizer_filename));
    }
    pad_id = processor.pad_id();
    vocab_size = processor.GetPieceSize();

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

  vector<int> random() {
    return read(runif(num()));
  }

  vector<int> read(int which) {
    if(which < 0 || which >= offsets.size()) {
      throw std::runtime_error("invalid which");
    }
    auto const& offset = offsets[which];
    file.seekg(offset, std::ios_base::beg);
    file.read(reinterpret_cast<char*>(&sz), sizeof(int32_t));
    string str(sz, ' ');
    file.read(str.data(), sz);
    vector<int> ret;
    auto const status = processor.Encode(str, &ret);
    if(!status.ok()) {
      throw std::runtime_error("could not encode from " + write_with_ss(offset));
    }
    if(ret.size() == 0) {
      throw std::runtime_error("empty number of tokens!");
    }
    return ret;
  }

  // Return the following (X,y) pair:
  //   (bsz x max_seq_len x vocab_size)
  //   (bsz x vocab_size)
  tuple<dbuffer_t, dbuffer_t>
  make_random_data(
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
      auto g = get_timetracker().make_full_gremlin("batch");
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

  int num() const { return offsets.size(); }
  int get_vocab_size() const { return vocab_size; }

private:
  using pos_type = std::ifstream::pos_type;
  std::ifstream file;
  sentencepiece::SentencePieceProcessor processor;
  vector<pos_type> offsets;
  int32_t sz;
  int pad_id;
  uint64_t vocab_size;
};

int main(int argc, char** argv) {
  args_t args(argc, argv);

  string tokenizer_file = args.get<string>("tokenizer");
  string samples_file   = args.get<string>("samples");

  dataset_reader_t reader(tokenizer_file, samples_file);

  int n = reader.num();
  for(int i = n-10; i != n; ++i) {
    DOUT(reader.read(i).size());
  }
  DOUT("BACK TO ZERO: " << reader.read(0).size());

  for(int i = 0; i != 2; ++i) {
    auto g = get_timetracker().make_full_gremlin("make_random_data");
    reader.make_random_data(dtype_t::f32, 2, 100000);
  }
  get_timetracker().print_totals(std::cout);
}
