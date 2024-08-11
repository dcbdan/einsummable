#include "../src/base/setup.h"

#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/simplescalarop.h"

int main()
{
    scalarop_t one = scalarop_t::make_constant(scalar_t::one(default_dtype()));
    scalarop_t none = scalarop_t::make_constant(scalar_t(default_dtype(), "-1.0"));
    scalarop_t add = scalarop_t::make_add();
    scalarop_t mul = scalarop_t::make_mul();
    scalarop_t sub = scalarop_t::make_sub();
    scalarop_t arg0 = scalarop_t::make_arg(0);
    scalarop_t arg1 = scalarop_t::make_arg(1);
    scalarop_t exp0 = scalarop_t::make_exp();

    {
        scalarop_t f = scalarop_t::replace_arguments(
            mul, {arg0, scalarop_t::replace_arguments(add, {one, arg0})});
        DOUT(f);
        list_simple_scalarop_t lop = list_simple_scalarop_t::make(f).value();
        DOUT(lop.to_scalarop());
        lop.print(std::cout);
    }

    DOUT("\n");

    {
        scalarop_t f = scalarop_t::replace_arguments(
            mul, {exp0, scalarop_t::replace_arguments(add, {one, exp0})});
        DOUT(f);
        list_simple_scalarop_t lop = list_simple_scalarop_t::make(f).value();
        DOUT(lop.to_scalarop());
        lop.print(std::cout);
    }

    DOUT("\n");

    {
        scalarop_t f =
            scalarop_t::combine(mul, {exp0, scalarop_t::replace_arguments(add, {one, exp0})});
        DOUT(f);
        list_simple_scalarop_t lop = list_simple_scalarop_t::make(f).value();
        DOUT(lop.to_scalarop());
        lop.print(std::cout);
    }

    DOUT("\n");

    {
        scalarop_t f = scalarop_t::combine(add, {add, add});
        f = scalarop_t::combine(f, {add, add, add, add});
        DOUT(f);
        list_simple_scalarop_t lop = list_simple_scalarop_t::make(f).value();
        DOUT(lop.to_scalarop());
        lop.print(std::cout);
    }

    DOUT("\n");

    {
        scalarop_t f = scalarop_t::make_relu();
        scalar_t   val(default_dtype(), "-9.0");
        f = scalarop_t::combine(f, {scalarop_t::make_scale(val)});
        DOUT(f);
        list_simple_scalarop_t lop = list_simple_scalarop_t::make(f).value();
        DOUT(lop.to_scalarop());
        lop.print(std::cout);
    }
}
