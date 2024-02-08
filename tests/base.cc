#include "test.h"

#include "partdim.h"
void add_tests_partdim(tests_t::context_t tests) {
  tests.insert_test("spans", test_partdim_spans);
  tests.insert_test("split", test_partdim_split);
}

int main() {
  tests_t tests("base");

  add_tests_partdim(tests.make_context("partdim"));

  tests.run();
}
