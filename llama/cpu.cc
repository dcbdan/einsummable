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

void time_contraction1() {
  dtype_t dtype = dtype_t::f16;

  // {8,256,4096,32,128}
  // {0,3,1,4},{2,3,4}->{0,1,2}

  {
    einsummable_t e(
      {8,256,4096,32,128},
      { {0,1,3,4}, {2,3,4} },
      3,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = build_einsummable(e.merge_adjacent_dims());
    gremlin_t timer("batch matmul " + write_with_ss(e));
    f(out_buffer.f16(), { lhs_buffer.f16(), rhs_buffer.f16() });
  }

  {
    // {8,256,4096,32,128}
    einsummable_t e(
      {8,256,4096,32,128},
      { {0,3,1,4}, {2,3,4} },
      3,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = contraction_t::make(
      dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
    buffer_t workspace = make_buffer(f.workspace_size);
    gremlin_t timer("contraction " + write_with_ss(e));
    f(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  }
}

void time_contraction2() {
  dtype_t dtype = dtype_t::f16;

  // ({0,2,1,4}->{0,1,2,4}),({0,3,1,4}->{0,1,4,3})->{0,1,2,3}
  // {8,32,256,256,128};

  {
    einsummable_t e(
      {8,32,256,256,128},
      { {0,1,2,4}, {0,1,4,3} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = build_einsummable(e.merge_adjacent_dims());
    gremlin_t timer("batch matmul " + write_with_ss(e));
    f(out_buffer.f16(), { lhs_buffer.f16(), rhs_buffer.f16() });
  }

  {
    einsummable_t e(
      {8,32,256,256,128},
      { {0,2,1,4}, {0,3,1,4} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = contraction_t::make(
      dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
    buffer_t workspace = make_buffer(f.workspace_size);
    gremlin_t timer("contraction " + write_with_ss(e));
    f(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  }
}

void time_contraction3() {
  dtype_t dtype = dtype_t::f16;

  // [8,32,256,128,256]
  // abce,aebd->abcd
  // 0124 0413  0123
  // bbij bjbk  bbik

  {
    einsummable_t e(
      {8,32,256,128,256},
      { {0,1,2,4}, {0,1,3,4} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = build_einsummable(e.merge_adjacent_dims());
    gremlin_t timer("batch matmul " + write_with_ss(e));
    f(out_buffer.f16(), { lhs_buffer.f16(), rhs_buffer.f16() });
  }

  {
    einsummable_t e(
      {8,32,256,128,256},
      { {0,1,2,4}, {0,4,1,3} },
      4,
      scalarop_t::make_mul(dtype),
      castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));
    dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

    lhs_buffer.random();
    rhs_buffer.random();
    auto f = contraction_t::make(
      dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
    buffer_t workspace = make_buffer(f.workspace_size);
    gremlin_t timer("contraction " + write_with_ss(e));
    f(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  }
}

void test_contraction1() {
  dtype_t dtype = dtype_t::f64;

  //einsummable_t e(
  //  {8,7,4,9,10},
  //  { {0,1,3,4}, {2,3,4} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);
  //einsummable_t e(
  //  {8,3,6,5,10},
  //  { {0,1,2,4}, {0,1,4,3} },
  //  4,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);
  einsummable_t e(
    {8,3,6,5,10},
    { {0,4,2,1}, {0,1,4,3} },
    4,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  auto inn_shapes = e.inn_shapes();
  dbuffer_t lhs_buffer = make_dbuffer(dtype, product(inn_shapes[0]));
  dbuffer_t rhs_buffer = make_dbuffer(dtype, product(inn_shapes[1]));

  lhs_buffer.random();
  rhs_buffer.random();

  dbuffer_t out_buffer_true =
    reference_einsummable(e, {lhs_buffer, rhs_buffer});

  dbuffer_t out_buffer = make_dbuffer(dtype, e.out_nelem());

  auto contraction = contraction_t::make(
    dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);
  buffer_t workspace = make_buffer(contraction.workspace_size);

  out_buffer.random();
  contraction(workspace->raw(), out_buffer.raw(), lhs_buffer.raw(), rhs_buffer.raw());
  if(is_close(out_buffer, out_buffer_true)) {
    DOUT("yes, is close");
  } else  {
    DOUT("IS NOT CLOSE!");
  }
}

void test_norm() {
  vector<double> data{
     0.104102411965,  1.75819599444 ,  0.241436410543, -0.923549600062,
     2.098498404957, -0.01980048358 ,  0.719057520776,  1.582565440747,
     0.011832471646,  0.583231115325,  1.735792867821, -1.075010469408,
    -0.491300759802,  0.900173883415, -0.696549140976, -0.017318511128,
     0.779691427962,  1.525656976835,  0.186386673303,  0.151327731509,
     0.372845878592,  0.132723343527,  1.466696600695, -0.293139876364,
    -0.575411014743, -0.661329473696, -0.112476311142, -0.681607609168,
     2.271264890919, -0.055723774135,  0.792926540523,  1.330586035915,
     0.014505599309,  1.072342068897,  0.502588730809, -0.034059581526,
     0.53419634156 , -1.864580855143,  0.944462033663, -0.153446086103
  };
  uint64_t ni = 2;
  uint64_t nj = 20;

  dtype_t dtype = dtype_t::f64;

  scalarop_t inverse_sqrt = scalarop_t::make_inverse_sqrt(dtype);
  scalarop_t square       = scalarop_t::make_square(dtype);
  scalarop_t identity     = scalarop_t::make_identity(dtype);

  scalar_t _e(dtype, write_with_ss(1e-6));
  scalar_t _a(dtype, write_with_ss(1.0/double(double(1.0)*nj)));
  scalarop_t scale_then_add_eps = scalarop_t::combine(
    scalarop_t::make_add(dtype),
    {
      scalarop_t::make_scale(_a),
      scalarop_t::make_constant(_e)
    });

  // y = np.power(np.mean(np.square(x), axis=-1) + eps, -0.5)

  einsummable_t e_square(
    {ni, nj},
    { {0,1} },
    2,
    square);
  einsummable_t e_reduction(
    {ni, nj},
    { {0, 1} },
    1,
    identity,
    castable_t::add);
  einsummable_t e_scale_add(
    {ni},
    { {0} },
    1,
    scale_then_add_eps);
  einsummable_t e_inverse_sqrt(
    {ni},
    { {0} },
    1,
    inverse_sqrt);

  dbuffer_t y = make_dbuffer(dtype, ni*nj);

  if(dtype == dtype_t::f16) {
    std::copy(data.begin(), data.end(), y.f16());
  } else if(dtype == dtype_t::f32) {
    std::copy(data.begin(), data.end(), y.f32());
  } else if(dtype == dtype_t::f64) {
    std::copy(data.begin(), data.end(), y.f64());
  }

  auto execute_einsummable = [](einsummable_t const& e, dbuffer_t data) {
    kernel_manager_t k;
    k.build(e);
    dbuffer_t out = make_dbuffer(e.out_dtype(), e.out_nelem());
    k(e, out.raw(), { data.raw() });
    return out;
  };

  //dbuffer_t y1 = reference_einsummable(e_square,       { y  });
  //dbuffer_t y2 = reference_einsummable(e_reduction,    { y1 });
  //dbuffer_t y3 = reference_einsummable(e_scale_add,    { y2 });
  //dbuffer_t y4 = reference_einsummable(e_inverse_sqrt, { y3 });

  dbuffer_t y1 = execute_einsummable(e_square,       y );
  dbuffer_t y2 = execute_einsummable(e_reduction,    y1);
  dbuffer_t y3 = execute_einsummable(e_scale_add,    y2);
  dbuffer_t y4 = execute_einsummable(e_inverse_sqrt, y3);

  auto print = [&dtype](string name, dbuffer_t y) {
    DOUT(name);
    if(y.dtype == dtype_t::f16) {
      for(int i = 0; i != y.nelem(); ++i) { DOUT(std::setprecision(12) << y.f16()[i]); }
    } else if(y.dtype == dtype_t::f32) {
      for(int i = 0; i != y.nelem(); ++i) { DOUT(std::setprecision(12) << y.f32()[i]); }
    } else if(y.dtype == dtype_t::f64) {
      for(int i = 0; i != y.nelem(); ++i) { DOUT(std::setprecision(12) << y.f64()[i]); }
    }
    DOUT("");
  };

  print("y1", y1);
  print("y2", y2);
  print("y3", y3);
  print("y4", y4);
  print("y5", y4.copy(dtype_t::f32));
}

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
  vtensor_t<int> tokens(
     {4,8},
     {1,  306, 4658,  278, 6593,  310, 2834, 338,
      1, 3439,17632, 1925,29892,  278, 6368, 310,
      1,17166,  263, 4700,  508,  367, 2309, 297,
      1, 4103, 9632, 4223,  304, 5176,29901,  13});

  uint64_t bsz    = tokens.get_shape()[0];
  uint64_t seqlen = tokens.get_shape()[1];

  auto args = model_args_t::llama_7B(bsz);

  builder_t build_first_token = builder_t::make_first_token(args, seqlen);

  // need all weights, freqcis, embeddings, mask

  map<int, buffer_t> data;
  for(auto const& [name, tinfo]: build_first_token.weights) {
    buffer_t w = reader(name);
    w = convert_f16_to_default(w);
    repartition_into_map_single_loc(data, tinfo, w);
  }

  if(build_first_token.mask) {
    auto const& tinfo = build_first_token.mask.value();
    buffer_t mask = transformer_t::form_start_mask(seqlen).data;
    repartition_into_map_single_loc(data, tinfo, mask);
  }

  {
    auto const& tinfo = build_first_token.freqs_cis;
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
      tokens).data;

    auto const& tinfo = build_first_token.embeddings;
    repartition_into_map_single_loc(data, tinfo, embeddings);
  }

  kernel_manager_t kernel_manager;

  for(auto const& node: build_first_token.taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& e = node.op.get_apply().einsummable;
      auto maybe = kernel_manager.build(e);
      if(!maybe) {
        throw std::runtime_error("could not build a kernel!");
      }
    }
  }

  settings_t settings {
    .num_apply_runner = 1,
    .num_touch_runner = 1, // subsets use touch kernels
    .num_send_runner = 0,
    .num_recv_runner = 0,
    .num_apply_kernel_threads = 12
  };

  execute(build_first_token.taskgraph, settings, kernel_manager, nullptr, data);

  dbuffer_t scores = unpartitioned_from_map_single_loc(data, build_first_token.scores);

  uint64_t top_n = 5;
  vtensor_t<int> top_choices = get_top_choices(
    scores, bsz, args.vocab_size, top_n);
  DOUT(top_choices.get());
}

int main(int argc, char** argv) {
  //float16_t x = scalar_t::negative_inf(dtype_t::f16).f16();
  //float16_t y(9.9);

  //DOUT(std::min(x,y));
  //DOUT(std::max(x,y));

  //DOUT(scalarop_t::make_min(dtype_t::f16).eval({scalar_t(x),scalar_t(y)}).f16());
  //DOUT(scalarop_t::make_max(dtype_t::f16).eval({scalar_t(x),scalar_t(y)}).f16());

  main_(argc, argv);

  //for(auto const& str: {"0.01", "0.1", "1.0", "10.0", "0.934"}) {
  //  DOUT(str << " " << scalar_t(dtype_t::f16, str).convert(dtype_t::f32));
  //}

  //test_norm();
}
