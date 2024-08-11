#include "test.h"

#include "partdim.h"
void add_tests_partdim(tests_t::context_t tests)
{
    tests.insert_test("spans", test_partdim_spans);
    tests.insert_test("split", test_partdim_split);
}

#include "reference.h"
void add_tests_reference(tests_t::context_t tests)
{
    tests.insert_test("repartition", test_reference_repartition);
    tests.insert_test("matmul", test_reference_matmul);
    tests.insert_test("reblock_20elem_01", reblock_20elem_test01);
    tests.insert_test("reblock_20elem_02", reblock_20elem_test02);
    tests.insert_test("reblock_20elem_03", reblock_20elem_test03);
}

int main()
{
    tests_t tests("base");

    add_tests_partdim(tests.make_context("partdim"));
    add_tests_reference(tests.make_context("reference"));

    tests.run();
}
