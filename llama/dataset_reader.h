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

  // Return the following (X,y) pair:
  //   (bsz x max_seq_len x vocab_size)
  //   (bsz x vocab_size)
  tuple<dbuffer_t, dbuffer_t>
  make_random_data(
    dtype_t dtype,
    uint64_t bsz,
    uint64_t max_seq_len);

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

