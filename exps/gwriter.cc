#include "../src/einsummable/graph.h"


void main01() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({100,10,200,10});
  x = x.view({1000,2000});

  tensor_t y = writer.input({200,10,300,10});
  y = y.view({2000,3000});

  tensor_t z = writer.matmul(x, y);
  z = z.view({1000,3000});

  writer.get_graph().print();
}

void main02() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({100,10,200,20});
  x = x.view({1000,4000});
  x = x.transpose(0, 1);
  DOUT(x.get_shape());
  x = x.view({200,20,100,10});
  DOUT(x.get_shape());

  x.save();
  writer.get_graph().print();
}

void main03() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});
  tensor_t y = writer.input({2,3,4,5});

  // this should not work
  writer.matmul(x, y);
}

void main04() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});
  tensor_t y = writer.input({2,3,6,5});

  tensor_t z = writer.matmul(x, y.transpose(2,3));

  DOUT(z.get_shape());
}

void main05() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});

  x = x.transpose(0,1);
  x = x.transpose(1,2);
  x = x.transpose(2,3);

  tensor_t z = writer.scale(88.0, x);
  z.save();

  tensor_t y = writer.add(x,x);

  y = writer.scale(9.9, y);
  y.save();

  writer.get_graph().print();
}

void main06() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});

  x = x.transpose(0,1);
  x = x.transpose(1,2);
  x = x.transpose(2,3);

  tensor_t z = writer.scale(88.0, x);
  z.save();

  tensor_t y = writer.add(x,x);

  y = writer.scale(9.9, y);
  y.save();

  writer.get_graph().print();
}

void main07() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});

  tensor_t y = writer.reduction("ijkl->ik", castable_t::add, x);

  x = x.view({6,20});
  tensor_t z = writer.reduction("ij->i", castable_t::min, x);

  y.save();
  z.save();

  writer.get_graph().print();
}

void main08_attention() {
  using tensor_t = graph_writer_t::tensor_t;
  graph_writer_t writer;

  uint64_t start_pos   = 0; // TODO: implement concat and set this
  uint64_t n_heads     = 8;
  uint64_t head_dim    = 10;
  uint64_t dim         = n_heads * head_dim;
  uint64_t bsz         = 11;
  uint64_t seqlen      = 12;

  vector<uint64_t> full_x = {bsz, seqlen, n_heads, head_dim};
  tensor_t x = writer.input(full_x).view(
    {bsz, seqlen, dim});

  vector<uint64_t> full_dd = {n_heads, head_dim, n_heads, head_dim};
  vector<uint64_t> dd      = {dim, dim};

  tensor_t wq = writer.input(full_dd).view(dd);
  tensor_t wk = writer.input(full_dd).view(dd);
  tensor_t wv = writer.input(full_dd).view(dd);
  tensor_t wo = writer.input(full_dd).view(dd);

  tensor_t xq = writer.matmul(x, wq.transpose(0,1));
  tensor_t xk = writer.matmul(x, wk.transpose(0,1));
  tensor_t xv = writer.matmul(x, wv.transpose(0,1));
  // xq/xk/xv: (bsz, seqlen, dim)

  xq = xq.view(full_x);
  xk = xk.view(full_x);
  xv = xv.view(full_x);
  // xq/xk/xv: (bsz, seqlen, heads, head dim)

  // TODO: rotary embed xq, xk

  //tensor_t prev_keys   = writer.input({bsz, start_pos, n_heads, head_dim});
  //tensor_t prev_values = writer.input({bsz, start_pos, n_heads, head_dim});
  //tensor_t keys   = writer.concat(1, {prev_keys,   xk});
  //tensor_t values = writer.concat(1, {prev_values, xv});
  tensor_t keys   = xk;
  tensor_t values = xv;

  xq     = xq.transpose(1, 2);
  keys   = keys.transpose(1, 2);
  values = values.transpose(1, 2);

  tensor_t scores = writer.matmul(xq, keys.transpose(2, 3));
  scores = writer.scale(1 / std::sqrt(1.0*head_dim), scores);
  // scores: bsz, n heads, seqlen, start pos + seqlen

  tensor_t mask = writer.input({bsz, n_heads, seqlen, start_pos + seqlen});
  scores = writer.add(scores, mask);

  scores = writer.softmax(scores);
  tensor_t output = writer.matmul(scores, values);
  output = output.transpose(1, 2).view({bsz, seqlen, dim});

  x = writer.matmul(output, wo.transpose(0,1));

  writer.get_graph().print();
}

int main() {
  main08_attention();
}

