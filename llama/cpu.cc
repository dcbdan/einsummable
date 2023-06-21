#include "misc.h"
#include "modules.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/kernels.h"

#include <fstream>

string replace_substrs(string text, string replace, string with)
{
  while(true) {
    auto pos = text.find(replace);
    if(text.npos != pos) {
      text.replace(pos, replace.size(), with);
    } else {
      break;
    }
  }

  return text;
}

void print_loop_kernel_info(bool unary, bool binary, bool keystr, bool opstr,
  bool keystrcd = false)
{
  vector<string> ks = {
    "ite_>=[constant{f32|0},hole|f32@0,constant{f32|0},hole|f32@0]",
    "power{2}[+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]]",
    "*[constant{f32|2},+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]]",
    "*[hole|f32@0,hole|f32@1]",
    "*[ite_>=[constant{f32|0},hole|f32@0,constant{f32|0},constant{f32|1}],hole|f32@1]",
    "+[hole|f32@0,*[constant{f32|-1},*[constant{f32|0.01},hole|f32@1]]]",
    "ite_>=[constant{f16|0},hole|f16@0,constant{f16|0},hole|f16@0]",
    "power{2}[+[hole|f16@0,*[constant{f16|-1},hole|f16@1]]]",
    "*[constant{f16|2},+[hole|f16@0,*[constant{f16|-1},hole|f16@1]]]",
    "*[hole|f16@0,hole|f16@1]",
    "*[ite_>=[constant{f16|0},hole|f16@0,constant{f16|0},constant{f16|1}],hole|f16@1]",
    "+[hole|f16@0,*[constant{f16|-1},*[constant{f16|1000},hole|f16@1]]]",
    "+[hole|f16@0,*[constant{f16|-1},*[constant{f16|1000},hole|f16@1]]]",
    "ite_>=[constant{f64|0},hole|f64@0,constant{f64|0},hole|f64@0]",
    "power{2}[+[hole|f64@0,*[constant{f64|-1},hole|f64@1]]]",
    "*[constant{f64|2},+[hole|f64@0,*[constant{f64|-1},hole|f64@1]]]",
    "*[hole|f64@0,hole|f64@1]",
    "*[ite_>=[constant{f64|0},hole|f64@0,constant{f64|0},constant{f64|1}],hole|f64@1]",
    "+[hole|f64@0,*[constant{f64|-1},*[constant{f64|1000.1},hole|f64@1]]]",
    "+[hole|f64@0,*[constant{f64|-1},*[constant{f64|1000.1},hole|f64@1]]]",
    "*[hole|f16@0,power{-1}[+[constant{f16|1},exp[*[constant{f16|-1},hole|f16@0]]]]]",
    "*[hole|f32@0,power{-1}[hole|f32@1]]",
    "exp[hole|f32@0]",
    "power{2}[hole|f32@0]",
    "+[*[constant{f32|0.00195312},hole|f32@0],constant{f32|1e-05}]",
    "power{-0.5}[hole|f32@0]",
    "to_f16[hole|f32@0]",
    "*[constant{f16|0.125},hole|f16@0]",
    "to_f32[hole|f16@0]",
    "+[*[constant{f32|0.00195312},hole|f32@0],constant{f32|1e-05}]",
    "+[hole|f16@0,hole|f16@1]",
    "+[hole|f32@0,hole|f32@1]"
  };

  auto to_type_str = [](dtype_t const& d) {
    if(d == dtype_t::f16) {
      return "float16_t";
    } else if(d == dtype_t::f32) {
      return "float";
    } else if(d == dtype_t::f64) {
      return "double";
    } else if(d == dtype_t::c64) {
      return "std::complex<float>";
    } else {
      throw std::runtime_error("should not reach");
    }
  };

  int nu = 0;
  int nb = 0;
  int nf = 0;
  for(auto const& s: ks) {
    scalarop_t f = parse_with_ss<scalarop_t>(s);
    auto [op_str, bytes] = f.to_cpp_bytes();
    string key = f.type_signature() + "|" + op_str;

    int n_inn = f.num_inputs();
    if(n_inn == 1) {
      if(unary) {
        if(keystr) {
         std::cout << "{ \"" << key <<
           "\", u" << nu << " }," << std::endl;
        }
        if(opstr) {
          auto tout = to_type_str(f.out_dtype());
          auto tinn = to_type_str(f.inn_dtype(0).value());
          std::cout << "_unary_ew_loop(u" << nu << ","
            << tout << "," << tinn << ","
            << op_str << ")" << std::endl;
        }
        nu++;
      }
    } else if(n_inn == 2) {
      if(binary) {
        if(keystr) {
         std::cout << "{ \"" << key <<
           "\", b" << nb << " }," << std::endl;
        }
        if(keystrcd) {
         std::cout << "{ \"" << key <<
           "\", { c" << nb << ", d" << nb << "} }," << std::endl;
        }
        if(opstr) {
          auto tout = to_type_str(f.out_dtype());
          auto tlhs = to_type_str(f.inn_dtype(0).value());
          auto trhs = to_type_str(f.inn_dtype(1).value());
          std::stringstream ss;
          ss << "_binary_ew_loop(b" << nb << ",c" << nb << ",d" << nb << ","
            << tout << "," << tlhs << "," << trhs << ","
            << op_str << ")";
          string str = replace_substrs(ss.str(), "x0[i]", "x0[i0]");
          str =        replace_substrs(str,      "x1[i]", "x1[i1]");
          std::cout << str << std::endl;
        }
        nb++;
      }
    } else {
      nf++;
    }
    //std::cout << std::endl;
  }

  if(nf != 0) {
    throw std::runtime_error("COULD NOT PROCESS ALL");
  }
}

void print_loop_kernel_info() {
  print_loop_kernel_info(true, false, true, false);
  print_loop_kernel_info(true, false, false, true);
  print_loop_kernel_info(false, true, true, false);
  print_loop_kernel_info(false, true, false, true);
  print_loop_kernel_info(false, true, false, false, true);
}

int main() {
  //print_loop_kernel_info();
  //return 1;

  set_default_dtype(dtype_t::f16);

  auto args = model_args_t::make_default();

  // TODO: set vocab_size
  args.vocab_size = 123;
  args.n_layers = 4;

  graph_writer_t writer;

  auto model = transformer_t(&writer, "name", args);

  uint64_t bsz = 3;
  uint64_t seq_len = 17;

  // read bsz and seq_len and input input tensor
  auto input_input = [&]() {
    full_shape_t shape({
      full_dim_t::singleton(bsz),
      full_dim_t::singleton(seq_len),
      args.full_dim()
    });

    return writer.input(shape);
  };

  tensor_t x = input_input();
  tensor_t y = model.forward(x);

  //seq_len = 1;

  //x = input_input();
  //y = model.forward(x);

  y = y.save();

  graph_t const& graph = writer.get_graph();

  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("wrote to g.gv");
  }

  auto const& [_0, _1, taskgraph] = taskgraph_t::make(
    graph,
    graph.make_singleton_placement());
  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("Printed to tg.gv");
  }

  DOUT("--------------");

  std::unordered_map<einsummable_t, kernel_t> kernels;
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto e = node.op.get_apply().einsummable.merge_adjacent_dims();
      if(kernels.count(e) == 0) {
        DOUT(e);
        DOUT(e.join);
        kernels[e] = build_einsummable(1, e);
        DOUT("---------------");
      }
    }
  }
  DOUT("done");
}
