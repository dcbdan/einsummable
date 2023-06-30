#include "../src/einsummable/graph.h"

void main01() {
  using tensor_t = graph_writer_t::tensor_t;
  using fs_t = graph_writer_t::full_shape_t;

  graph_writer_t writer;

  tensor_t x = writer.input(
    fs_t::from_vecvec({{100,10},{200,10}}));

  tensor_t y = writer.input({200,10,300,10});
  y = y.view({{200,10},{300,10}});

  tensor_t z = writer.matmul(x, y);
  z = z.view_full();

  writer.get_graph().print();
}

void main02() {
  using tensor_t = graph_writer_t::tensor_t;

  using fd_t = graph_writer_t::full_dim_t;
  using fs_t = graph_writer_t::full_shape_t;

  fd_t di({100,10});
  fd_t dj({200,20});

  graph_writer_t writer;

  tensor_t x = writer.input({100,10,200,20});
  x = x.view(fs_t({di,dj}));
  x = x.transpose(0, 1);
  DOUT(x.get_shape());
  x = x.view_full({200,20,100,10});
  DOUT(x.get_shape());

  x = x.save();
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

  tensor_t z = writer.scale("88.0", x).save();

  tensor_t y = writer.add(x,x);

  y = writer.scale("9.9", y).save();

  writer.get_graph().print();
}

void main06() {
  using tensor_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});

  x = x.transpose(0,1);
  x = x.transpose(1,2);
  x = x.transpose(2,3);

  tensor_t z = writer.scale("88.0", x);
  z = z.save();

  tensor_t y = writer.add(x,x);

  y = writer.scale("9.9", y).save();

  writer.get_graph().print();
}

void main07() {
  using tensor_t = graph_writer_t::tensor_t;

  using fd_t = graph_writer_t::full_dim_t;
  using fs_t = graph_writer_t::full_shape_t;

  graph_writer_t writer;

  tensor_t x = writer.input({2,3,4,5});

  tensor_t y = writer.reduction("ijkl->ik", castable_t::add, x);

  fd_t di({2,3});
  fd_t dj({4,5});
  x = x.view(fs_t({di,dj}));

  tensor_t z = writer.reduction("ij->i", castable_t::min, x);

  y = y.save();
  z = z.save();

  writer.get_graph().print();
}

void main08_attention() {
  using tensor_t = graph_writer_t::tensor_t;

  using fd_t = graph_writer_t::full_dim_t;
  using fs_t = graph_writer_t::full_shape_t;

  graph_writer_t writer;

  fd_t start_pos({ 7 });
  fd_t n_heads  ({ 8 });
  fd_t head_dim ({ 10 });
  fd_t dim      ({ n_heads(), head_dim() });
  fd_t bsz      ({ 11 });
  fd_t seqlen   ({ 12 });

  fs_t full_x({bsz, seqlen, n_heads, head_dim});
  tensor_t x = writer.input(full_x).view(fs_t({bsz, seqlen, dim}));

  fs_t full_dd({n_heads, head_dim, n_heads, head_dim});
  fs_t dd({dim, dim});

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

  tensor_t prev_keys   = writer.input(fs_t({bsz, start_pos, n_heads, head_dim}));
  tensor_t prev_values = writer.input(fs_t({bsz, start_pos, n_heads, head_dim}));
  tensor_t keys   = writer.concat(1, {prev_keys,   xk});
  tensor_t values = writer.concat(1, {prev_values, xv});

  xq     = xq.transpose(1, 2);
  keys   = keys.transpose(1, 2);
  values = values.transpose(1, 2);

  tensor_t scores = writer.matmul(xq, keys.transpose(2, 3));
  string _scale_val = write_with_ss(double(1 / std::sqrt(1.0*head_dim())));
  scores = writer.scale(_scale_val, scores);
  // scores: bsz, n heads, seqlen, start pos + seqlen

  tensor_t mask = writer.input({bsz(), n_heads(), seqlen(), start_pos() + seqlen()});
  scores = writer.add(scores, mask);

  scores = writer.softmax(scores);
  tensor_t output = writer.matmul(scores, values);
  output = output.transpose(1, 2).view(fs_t({bsz, seqlen, dim}));

  x = writer.matmul(output, wo.transpose(0,1));

  writer.get_graph().print();
}

void main09_subset() {
  graph_writer_t writer;

  using tensor_t = graph_writer_t::tensor_t;
  using fd_t = graph_writer_t::full_dim_t;
  using fs_t = graph_writer_t::full_shape_t;

  using _all = graph_writer_t::idx_t::all;
  using _idx = graph_writer_t::idx_t::idx;
  using _rng = graph_writer_t::idx_t::rng;

  fd_t di({20});
  fd_t dj({30});
  fd_t dk({40,50});

  tensor_t a = writer.input(fs_t({di,dj,dk}));

  tensor_t b = a.subset({ _all{}, _all{}, _all{}});
  DOUT(b.get_shape() << " | " << a.get_id() << " " << b.get_id());
  // b is just a no op

  tensor_t c = a.subset({ _rng{5,10}, _rng{10,15}, _all{} });
  DOUT(c.get_shape());

  tensor_t d = a.subset({ _rng{5,10}, _idx{9}, _all{} });
  DOUT(d.get_shape());

  writer.get_graph().print();

  // this should fail
  //tensor_t e = a.subset({ _all{}, _all{}, _rng{3,6} });
}

int main() {
  main09_subset();
}

