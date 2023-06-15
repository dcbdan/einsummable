#include "misc.h"
#include "modules.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

using tensor_t = graph_writer_t::tensor_t;

void rms_norm_test() {
  dtype_t dtype = dtype_t::f16;
  set_default_dtype(dtype);

  auto [buffer_w, shape_w] = read_array(dtype,
    "[0.4661, 0.9641, 0.3547, 0.0875, 0.7970]");
  auto [buffer_x, shape_x] = read_array(dtype,
    "[[[0.0333, 0.0559, 0.7846, 0.7317, 0.6389],    "
    "     [0.5292, 0.5690, 0.5824, 0.7414, 0.8861], "
    "     [0.3719, 0.0415, 0.2344, 0.7994, 0.4197], "
    "     [0.3989, 0.5676, 0.8946, 0.9538, 0.0057]],"
    "    [[0.8891, 0.3468, 0.7371, 0.6944, 0.7015], "
    "     [0.9771, 0.7700, 0.5610, 0.5581, 0.0082], "
    "     [0.8135, 0.8535, 0.5942, 0.2557, 0.0224], "
    "     [0.4991, 0.5930, 0.1843, 0.4931, 0.5189]],"
    "    [[0.6224, 0.0205, 0.0650, 0.1405, 0.9574], "
    "     [0.3674, 0.7396, 0.6307, 0.2756, 0.8961], "
    "     [0.3821, 0.8173, 0.3014, 0.9887, 0.1508], "
    "     [0.1859, 0.1640, 0.6373, 0.1771, 0.4573]]]");
  auto [buffer_y, shape_y] = read_array(dtype,
    "[[[0.0277, 0.0965, 0.4977, 0.1145, 0.9106],    "
    "     [0.3654, 0.8128, 0.3061, 0.0961, 1.0464], "
    "     [0.3856, 0.0890, 0.1850, 0.1556, 0.7443], "
    "     [0.2808, 0.8266, 0.4793, 0.1260, 0.0068]],"
    "    [[0.5946, 0.4798, 0.3752, 0.0871, 0.8023], "
    "     [0.6906, 1.1259, 0.3018, 0.0740, 0.0099], "
    "     [0.6303, 1.3680, 0.3504, 0.0372, 0.0297], "
    "     [0.4856, 1.1936, 0.1365, 0.0900, 0.8635]],"
    "    [[0.5628, 0.0383, 0.0447, 0.0238, 1.4804], "
    "     [0.2736, 1.1393, 0.3574, 0.0385, 1.1411], "
    "     [0.2885, 1.2765, 0.1732, 0.1401, 0.1947], "
    "     [0.2302, 0.4203, 0.6007, 0.0412, 0.9684]]]");

  uint64_t d = 5;
  graph_writer_t writer;
  float eps = 1e-6;
  rms_norm_t rms_norm(&writer, "name", full_dim_t{ {d} }, eps);

  tensor_t x = writer.input(shape_x, dtype);
  tensor_t y = rms_norm.forward(x);

  y.save();

  map<int, dbuffer_t> inputs {
    { x.get_id(), buffer_x },
    { rms_norm.weight.get_id(), buffer_w }
  };

  auto saves = reference_compute_graph(writer.get_graph(), inputs);
  dbuffer_t _buffer_y = saves.at(y.get_id());

  DOUT(buffer_y);
  DOUT(_buffer_y);
}

int main() {
//  set_default_dtype(dtype_t::f16);
//
//  auto model_args = model_args_t::make_default();
//
//  graph_writer_t writer;
//
//  auto model = transformer_t(writer, "name", model_args, 1);
}


