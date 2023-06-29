#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/kernels.h"
#include "../src/execution/cpu/permute.h"
#include "../src/execution/cpu/contraction.h"

#include "../src/execution/cpu/execute.h"

#include <fstream>

#include <mkl.h> // for mkl_set_num_threads

#include <iomanip> // setprecision

struct tensor_reader_t {
  tensor_reader_t(string filename) : file(filename, std::ios::binary) {
    if(!file) {
      throw std::runtime_error("Failed to open the file.");
    }
  }

  vector<string> all_names() {
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

  buffer_t operator()(string tensor_name) {
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

private:
  std::ifstream file;

  void to_beginning() {
    file.clear();
    file.seekg(0);
  }

  optional<tuple<string, uint64_t>>
  read_next_weight_info() {
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
};

struct token_maker_t {
  token_maker_t(vector<vector<int>> const ps):
    prompts(ps)
  {
    int bsz = prompts.size();
    if(bsz == 0) {
      throw std::runtime_error("must give atleast one prompt");
    }

    int seqlen = prompts[0].size();
    for(auto const& prompt: prompts) {
      seqlen = std::min(std::size_t(seqlen), prompt.size());
    }
    if(seqlen == 0) {
      throw std::runtime_error("cannot have empty prompt");
    }

    tokens = vtensor_t<int>({bsz, seqlen});
    for(int i = 0; i != bsz;    ++i) {
    for(int j = 0; j != seqlen; ++j) {
      tokens.at({i,j}) = prompts[i][j];
    }}
  }

  int batch_size() const {
    return tokens.get_shape()[0];
  }

  vtensor_t<int> const& get_tokens() const {
    return tokens;
  }

  vtensor_t<int> operator()(int start_pos, int end_pos) const {
    auto shape = tokens.get_shape();
    return tokens.subset({ {0, shape[0]}, {start_pos, end_pos} });
  }

  vtensor_t<int> last_column() const {
    auto shape = tokens.get_shape();
    return this->operator()(shape[1]-1, shape[1]);
  }

  void add_next_tokens(vtensor_t<int> const& next_tokens) {
    int startpos = tokens.get_shape()[0];

    tokens = vtensor_t<int>::concat(1, {tokens, next_tokens});

    // now apply the mask
    auto shape = tokens.get_shape();

    int bsz    = shape[0];
    int seqlen = shape[1] - startpos;

    for(int i = 0; i != bsz; ++i) {
      for(int j = startpos; j < prompts[i].size() && j < shape[1]; ++j) {
        tokens.at({i,j}) = prompts[i][j];
      }
    }
  }

private:
  vector<vector<int>> prompts;
  vtensor_t<int> tokens;
};

dbuffer_t lookup_embeddings(
  int nvocab,
  int nembed,
  dbuffer_t const& data,
  vtensor_t<int> tokens)
{
  if(data.nelem() != nvocab * nembed) {
    throw std::runtime_error("incorrectly sized data matrix");
  }

  int ntokens = product(tokens.get_shape());
  dbuffer_t ret = make_dbuffer(data.dtype, nembed * ntokens);

  char const* data_raw = reinterpret_cast<char const*>(data.raw());
  char      * ret_raw  = reinterpret_cast<char      *>(ret.raw());

  uint64_t stride = dtype_size(data.dtype) * nembed;

  for(int i = 0; i != ntokens; ++i) {
    int const& token = tokens.get()[i];
    if(token < 0 || token >= nvocab) {
      throw std::runtime_error("invalid token");
    }
    std::copy(
      data_raw + token    *stride,
      data_raw + (token+1)*stride,
      ret_raw  + i        *stride);
  }

  return ret;
}

template <typename T>
vtensor_t<int> _get_top_choices(
  T const* data,
  uint64_t nrow,
  uint64_t ncol,
  uint64_t topn)
{
  topn = std::min(topn, ncol);
  vtensor_t<int> ret({int(nrow), int(topn)});
  int* ret_vec = ret.get().data();
  for(int i = 0; i != nrow; ++i) {
    T const* d = data    + i*ncol;
    int*     r = ret_vec + i*topn;
    vector<T const*> tops = select_topn(d, d + ncol, topn);
    for(int j = 0; j != topn; ++j) {
      r[j] = std::distance(d, tops[j]);
    }
  }
  return ret;
}

vtensor_t<int> get_top_choices(
  dbuffer_t const& data,
  uint64_t nrow,
  uint64_t ncol,
  uint64_t topn)
{
  if(data.dtype == dtype_t::f16) {
    return _get_top_choices(data.f16(), nrow, ncol, topn);
  } else if(data.dtype == dtype_t::f32) {
    return _get_top_choices(data.f32(), nrow, ncol, topn);
  } else if(data.dtype == dtype_t::f64) {
    return _get_top_choices(data.f64(), nrow, ncol, topn);
  }
  throw std::runtime_error("get_top_choices: no dtype support here");
}

vtensor_t<int> append_tokens_with_best(
  vtensor_t<int> const& prev_tokens, // batch by seqlen
  vtensor_t<int> const& best_n) // batch by topn ; only use the first column
{
  auto block_shape = prev_tokens.get_shape();
  int bsz = block_shape[0];
  int seqlen = block_shape[1];

  vtensor_t<int> ret({bsz, seqlen+1});
  for(int b = 0; b != bsz;    ++b) {
    int s = 0;
    for(s = 0; s != seqlen; ++s) {
      ret.at({b,s}) = prev_tokens.at({b,s});
    }
    ret.at({b,s}) = best_n.at({b,0});
  }

  return ret;
}

void main_(int argc, char** argv) {
  if(argc != 2) {
    throw std::runtime_error("usage: filename");
  }

  set_default_dtype(dtype_t::f16);

  tensor_reader_t reader(argv[1]);

  auto convert_f16_to_default = [](buffer_t buffer) {
    if(default_dtype() != dtype_t::f16) {
      buffer = dbuffer_t(dtype_t::f16, buffer).copy(default_dtype()).data;
    }
    return buffer;
  };

  // prompts = [
  //   # For these prompts, the expected answer is the natural continuation of the prompt
  //   "I believe the meaning of life is",
  //   "Simply put, the theory of relativity states that ",
  //   "Building a website can be done in 10 simple steps:\n",
  //   """Translate English to French:
  //      sea otter => loutre de mer
  //      plush girafe => girafe peluche
  //      cheese =>"""
  // ]
  token_maker_t token_maker({
    {1, 306, 4658, 278, 6593, 310, 2834, 338},
    {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871},
    {1, 17166, 263, 4700, 508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29901, 13},
    {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 268, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 268, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 268, 923, 968, 1149}
  });

  vtensor_t<int> init_tokens = token_maker.get_tokens();
  DOUT(init_tokens.get());

  uint64_t bsz    = init_tokens.get_shape()[0];
  uint64_t seqlen = init_tokens.get_shape()[1];

  auto args = model_args_t::llama_7B(bsz);

  int niter = 10;

  builder_t builder = builder_t::make_first_token(args, seqlen);

  // need all weights, freqcis, embeddings, mask

  map<int, buffer_t> data;
  for(auto const& [name, tinfo]: builder.weights) {
    buffer_t w = reader(name);
    w = convert_f16_to_default(w);
    repartition_into_map_single_loc(data, tinfo, w);
  }

  if(builder.mask) {
    auto const& tinfo = builder.mask.value();
    buffer_t mask = transformer_t::form_start_mask(seqlen).data;
    repartition_into_map_single_loc(data, tinfo, mask);
  }

  // dies here?
  {
    auto const& tinfo = builder.freqs_cis;
    buffer_t freqs_cis = transformer_t::form_full_freqs_cis(args).data;
    repartition_into_map_single_loc(data, tinfo, freqs_cis);
  }

  dbuffer_t embedding_matrix(
    default_dtype(),
    convert_f16_to_default(reader("tok_embeddings.weight")));

  {
    buffer_t embeddings = lookup_embeddings(
      args.vocab_size,
      args.dim,
      embedding_matrix,
      init_tokens).data;

    auto const& tinfo = builder.embeddings;
    repartition_into_map_single_loc(data, tinfo, embeddings);
  }

  kernel_manager_t kernel_manager;

  for(auto const& node: builder.taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& e = node.op.get_apply().einsummable;
      auto maybe = kernel_manager.build(e);
      if(!maybe) {
        throw std::runtime_error("could not build a kernel!");
      }
    }
  }

  settings_t settings {
    .num_apply_runner = 12,
    .num_touch_runner = 4, // subsets use touch kernels
    .num_send_runner = 0,
    .num_recv_runner = 0,
    .num_apply_kernel_threads = 1
  };

  DOUT("---");
  execute(builder.taskgraph, settings, kernel_manager, nullptr, data);

  //for(auto const& [tid, buffer]: data) {
  //  DOUT(tid << "  " << dbuffer_t(dtype_t::f16, buffer).sum_to_f64());
  //}

  {
    dbuffer_t scores = unpartitioned_from_map_single_loc(data, builder.scores);

    uint64_t top_n = 5;
    vtensor_t<int> top_choices = get_top_choices(
      scores, bsz, args.vocab_size, top_n);
    DOUT(top_choices.get());

    token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
  }

  for(int i = 0; i != niter; ++i) {
    builder = builder_t::make_next_token(builder, (i + 1 == niter));

    //builder.print_info();

    builder.transform_from_prev(data);

    {
      buffer_t embeddings = lookup_embeddings(
        args.vocab_size,
        args.dim,
        embedding_matrix,
        token_maker.last_column()).data;

      auto const& tinfo = builder.embeddings;
      repartition_into_map_single_loc(data, tinfo, embeddings);
    }

    for(auto const& node: builder.taskgraph.nodes) {
      if(node.op.is_apply()) {
        auto const& e = node.op.get_apply().einsummable;
        auto maybe = kernel_manager.build(e);
        if(!maybe) {
          throw std::runtime_error("could not build a kernel!");
        }
      }
    }

    DOUT("---");
    execute(builder.taskgraph, settings, kernel_manager, nullptr, data);

    {
      dbuffer_t scores = unpartitioned_from_map_single_loc(data, builder.scores);

      uint64_t top_n = 5;
      vtensor_t<int> top_choices = get_top_choices(
        scores, bsz, args.vocab_size, top_n);
      DOUT(top_choices.get());

      token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
    }
  }

  DOUT(token_maker.get_tokens().get());
}

int main(int argc, char** argv) {
  main_(argc, argv);
}
