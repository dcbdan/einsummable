#include "../src/base/setup.h"

#include <fstream>

#include "../src/einsummable/dbuffer.h"
#include "../src/base/indexer.h"

#include "piper.h"

struct just_tokenizer_t {
    just_tokenizer_t(string tokenizer_filename);

    vector<int> operator()(string const& msg);

    int const& pad_id() const
    {
        return pad_id_;
    }
    uint64_t const& vocab_size() const
    {
        return vocab_size_;
    }

private:
    piper_t  piper;
    int      pad_id_;
    uint64_t vocab_size_;
};

struct dataset_reader_t {
    dataset_reader_t(string tokenizer_filename, string data_filename);

    vector<int> random()
    {
        return read(runif(num()));
    }

    vector<int> read(int which);

    // NOTE: this method still has a random component, because it has
    //       to select somewhere
    tuple<vector<int>, int> datum(int which, uint64_t seqlen);

    // return a seqlen vector and the seqlen
    tuple<vector<int>, int> random_datum(uint64_t seqlen);

    tuple<vector<vector<int>>, vector<int>> random_data(uint64_t batch_size, uint64_t seqlen);

    // return (tokens.size(), embed size)
    dbuffer_t make_embedding(dbuffer_t const&   embedding_matrix, // (vocab size, embed size)
                             vector<int> const& tokens);

    // (batch size, seq_len, vocab size)
    dbuffer_t one_hot_encode(dtype_t dtype, vector<int> const& tokens);

    int num() const
    {
        return offsets.size();
    }
    int get_vocab_size() const
    {
        return vocab_size;
    }
    int get_pad_id() const
    {
        return pad_id;
    }

private:
    just_tokenizer_t tokenizer;

    using pos_type = std::ifstream::pos_type;
    std::ifstream    file;
    vector<pos_type> offsets;
    int32_t          sz;
    int              pad_id;
    uint64_t         vocab_size;
};

// return (tokens.size(), embed size)
dbuffer_t make_embedding(uint64_t           vocab_size,
                         dbuffer_t const&   embedding_matrix, // (vocab size, embed size)
                         vector<int> const& tokens);
