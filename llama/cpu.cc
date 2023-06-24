#include "misc.h"
#include "modules.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/kernels.h"
#include "../src/execution/cpu/permute.h"
#include "../src/execution/cpu/contraction.h"

#include <fstream>

#include <mkl.h> // for mkl_set_num_threads

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

map<string, buffer_t>
read_all_weights(string filename)
{
  std::ifstream file(filename, std::ios::binary);

  if(!file) {
    throw std::runtime_error("Failed to open the file.");
  }

  map<string, buffer_t> extracted_data;

  while(true) {
    // Read the text data (name of weight tensor)
    char text_data[51];
    file.read(text_data, 50);
    text_data[50] = '\0';
    std::string name(text_data);

    // If no more data to read, break the loop
    if(name.empty() || file.eof()) {
      break;
    }

    std::string space = " ";
    const auto strEnd = name.find_last_not_of(space);
    const auto strRange = strEnd + 1;
    name = name.substr(0, strRange);

    // Read the binary data (size of tensor)
    int64_t nelem;
    file.read(reinterpret_cast<char*>(&nelem), sizeof(int64_t));

    // Assuming all tensors are stored in float16s
    uint64_t size = nelem * sizeof(float16_t);

    // Read the tensor data
    buffer_t buffer = make_buffer(size);
    file.read(reinterpret_cast<char*>(buffer->data), size);

    // Store the extracted data
    extracted_data.insert({name, buffer});
  }

  return extracted_data;
}

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


void main_(int argc, char** argv) {
  if(argc != 2) {
    throw std::runtime_error("usage: filename");
  }

  set_default_dtype(dtype_t::f16);

  auto args = model_args_t::llama_7B();

  graph_writer_t writer;
  auto model = transformer_t(&writer, "name", args);

  buffer_t embedding_matrix;
  map<int, buffer_t> graph_inputs;
  {
    map<string, buffer_t> weights = read_all_weights(argv[1]);

    embedding_matrix = weights.at("tok_embeddings.weight");

    std::set<string> from_model;
    for(auto const& [id, name]: model.input_map()) {
      graph_inputs.insert({id, weights.at(name)});
      from_model.insert(name);
    }

    for(auto const& [name, _]: weights) {
      if(from_model.count(name) == 0) {
        if(name.find("freqs") != string::npos || name == "tok_embeddings.weight") {
          // these are not necc
        } else {
          throw std::runtime_error("missing weight: " + name);
        }
      }
    }
  }

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
      1, 4103, 9632, 4223,  304, 5176,29901,  13 });

  uint64_t bsz    = tokens.get_shape()[0];
  uint64_t seqlen = tokens.get_shape()[1];

  dbuffer_t x_data = lookup_embeddings(
    args.vocab_size,
    args.dim,
    dbuffer_t(dtype_t::f16, embedding_matrix),
    tokens);

  tensor_t x = writer.input(full_shape_t({
    full_dim_t::singleton(bsz),
    full_dim_t::singleton(seqlen),
    args.full_dim()
  }));

  tensor_t y = model.forward(x);

  y = y.save();

  graph_t const& graph = writer.get_graph();

  graph_inputs.insert({ x.get_id(), x_data.data });

  graph_inputs.insert({
    model.full_freqs_cis.get_id(),
    model.form_full_freqs_cis(args).data
  });

  graph_inputs.insert({
    model.mask_infos.back().mask.get_id(),
    model.form_start_mask(seqlen).data
  });

  //{
  //  std::ofstream f("g.gv");
  //  graph.print_graphviz(f);
  //  DOUT("wrote to g.gv");
  //}

  auto const& [_0, _1, taskgraph] = taskgraph_t::make(
    graph,
    graph.make_singleton_placement());
  //{
  //  std::ofstream f("tg.gv");
  //  taskgraph.print_graphviz(f);
  //  DOUT("Printed to tg.gv");
  //}

  kernel_manager_t kernel_manager;
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& e = node.op.get_apply().einsummable;
      auto maybe = kernel_manager.build(e);
      if(!maybe) {
        throw std::runtime_error("could not build a kernel!");
      }
    }
  }
}

int main(int argc, char** argv) {
  main_(argc, argv);
}
