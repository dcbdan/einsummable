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

int main() {
  main07();
}


