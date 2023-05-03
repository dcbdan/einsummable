#include "../src/execution/cpu/kernels.h"

#include "../src/einsummable/reference.h"

bool valid_touchdim(touchdim_t const& t) {
  if(t.d_inn == 0 || t.d_out == 0 || t.size == 0) {
    return false;
  }
  if(t.size + t.offset_inn > t.d_inn) {
    return false;
  }
  if(t.size + t.offset_out > t.d_out) {
    return false;
  }
  return true;
}

touchdim_t random_touchdim() {
  uint64_t d1 = runif(10, 30);
  uint64_t d2 = runif(10, 30);
  uint64_t sz = runif(1, std::min(d1, d2));
  uint64_t o1 = runif(0, d1-sz);
  uint64_t o2 = runif(0, d2-sz);

  touchdim_t ret {
    .d_inn = d1,
    .d_out = d2,
    .offset_inn = o1,
    .offset_out = o2,
    .size = sz
  };

  if(!valid_touchdim(ret)) {
    throw std::runtime_error("check random touchdim...");
  }

  return ret;
}

optional<castable_t> random_castable() {
  int i = runif(-1, 4);
  if(i == -1) {
    return optional<castable_t>();
  }
  if(i == 0) {
    return optional<castable_t>(castable_t::add);
  }
  if(i == 1) {
    return optional<castable_t>(castable_t::mul);
  }
  if(i == 2) {
    return optional<castable_t>(castable_t::min);
  }
  if(i == 3) {
    return optional<castable_t>(castable_t::max);
  }
  throw std::runtime_error("should not reach");
}

touch_t random_touch(int sz) {
  vector<touchdim_t> selection;
  for(int i = 0; i != sz; ++i) {
    selection.push_back(random_touchdim());
  }
  return touch_t {
    .selection = selection,
    .castable = random_castable()
  };
}

void test_touch(touch_t const& touch) {
  vector<uint64_t> shape_out = vector_from_each_member(touch.selection, uint64_t, d_out);
  vector<uint64_t> shape_inn = vector_from_each_member(touch.selection, uint64_t, d_inn);

  uint64_t sz_out = product(shape_out);
  uint64_t sz_inn = product(shape_inn);

  buffer_t out1 = std::make_shared<buffer_holder_t>(sz_out);
  buffer_t out2 = std::make_shared<buffer_holder_t>(sz_out);
  buffer_t out3 = std::make_shared<buffer_holder_t>(sz_out);
  buffer_t inn  = std::make_shared<buffer_holder_t>(sz_inn);

  out1->zeros();
  out2->zeros();
  inn->random(-2.0, 2.0);

  {
    raii_print_time_elapsed_t gremlin("reference   ");
    reference_touch(touch, out1, inn);
  }

  {
    auto built = build_touch(touch);
    raii_print_time_elapsed_t gremlin("built       ");
    built(out2->data, inn->data);
  }

  if(out1 != out2) {
    throw std::runtime_error("They are not correct!");
  }

}

int main() {
  int n = 20;

  std::cout << "dim 1" << std::endl;
  for(int i = 0; i != n; ++i) {
    test_touch(random_touch(1));
  }

  std::cout << "dim 2" << std::endl;
  for(int i = 0; i != n; ++i) {
    test_touch(random_touch(2));
  }

  std::cout << "dim 3" << std::endl;
  for(int i = 0; i != n; ++i) {
    test_touch(random_touch(3));
  }

  std::cout << "dim 4" << std::endl;
  for(int i = 0; i != n; ++i) {
    test_touch(random_touch(4));
  }
}