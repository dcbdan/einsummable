#include "../src/base/setup.h"

#include <fstream>

#include "../src/einsummable/dbuffer.h"
#include "../src/base/indexer.h"

#include "piper.h"

struct dataset_reader_t {
  dataset_reader_t(string tokenizer_filename, string data_filename);

  vector<int> random() {
    return read(runif(num()));
  }

  vector<int> read(int which);

  // return a seqlen vector and the seqlen
  tuple<vector<int>, int> random_datum(uint64_t seqlen);

  tuple<vector<vector<int>>, vector<int>>
  random_data(uint64_t batch_size, uint64_t seqlen);

  // return (tokens.size(), embed size)
  dbuffer_t make_embedding(
    dbuffer_t const& embedding_matrix, // (vocab size, embed size)
    vector<int> const& tokens);

  // (batch size, seq_len, vocab size)
  dbuffer_t one_hot_encode(dtype_t dtype, vector<int> const& tokens);

  int num() const { return offsets.size(); }
  int get_vocab_size() const { return vocab_size; }

private:
  using pos_type = std::ifstream::pos_type;
	piper_t piper;
  std::ifstream file;
  vector<pos_type> offsets;
  int32_t sz;
  int pad_id;
  uint64_t vocab_size;
};

