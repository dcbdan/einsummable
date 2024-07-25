#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/dbuffer.h"

#include "../src/execution/cpu/executetg.h"

#include <fstream>

#include "tblis/tblis.h"

using tensor_t = graph_writer_t::tensor_t;
using full_dim_t = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

graph_t
make_attention_graph(uint64_t batch_size, uint64_t start_pos, uint64_t dim, uint64_t n_heads)
{
    graph_writer_t writer;

    uint64_t head_dim = uint64_div(dim, n_heads, "head_dim");

    full_dim_t   full_dim({n_heads, head_dim});
    full_shape_t kqv_shape({full_dim, full_dim});

    tensor_t wq = writer.input(kqv_shape);
    // tensor_t wk = writer.input(kqv_shape);
    // tensor_t wv = writer.input(kqv_shape);
    tensor_t wo = writer.input(kqv_shape);

    vector<uint64_t> prev_shape{batch_size, start_pos, n_heads, head_dim};

    tensor_t keys = writer.input(prev_shape);
    tensor_t values = writer.input(prev_shape);

    full_shape_t     xshape(vector<full_dim_t>(
        {full_dim_t::singleton(batch_size), full_dim_t::singleton(1), full_dim}));
    vector<uint64_t> full_xshape = xshape.full();

    tensor_t x = writer.input(xshape);

    tensor_t xq = writer.matmul(x, wq.transpose(0, 1));
    // tensor_t xk = writer.matmul(x, wk.transpose(0,1));
    // tensor_t xv = writer.matmul(x, wv.transpose(0,1));

    xq = xq.view_full(full_xshape);
    // xk = xk.view_full(full_xshape);
    // xv = xv.view_full(full_xshape);

    xq = xq.transpose(1, 2);
    keys = keys.transpose(1, 2);
    values = values.transpose(1, 2);

    scalarop_t scale = scalarop_t::make_scale(
        scalar_t(default_dtype(), write_with_ss(1.0 / (std::sqrt(double(1.0) * head_dim)))));

    tensor_t scores;
    scores = writer.matmul(xq, keys.transpose(2, 3));
    // scores = writer.ew(scale, scores);
    // scores = writer.softmax(scores);

    tensor_t output;
    output = writer.matmul(scores, values);
    output = output.transpose(1, 2);
    output = output.view(xshape);
    output = writer.matmul(output, wo.transpose(0, 1));

    return writer.get_graph();
}

einsummable_t make_e(vector<uint64_t> join_shape, string str)
{
    auto [inns, out_rank] = einsummable_t::parse_str(str);
    return einsummable_t(join_shape, inns, out_rank, scalarop_t::make_mul(), castable_t::add);
}

buffer_t make_data(vector<uint64_t> const& shape)
{
    auto d = make_dbuffer(dtype_t::f32, product(shape));
    d.random("-0.00001", "0.00001");
    return d.data;
}

buffer_t make_data(einsummable_t const& e, int which_inn)
{
    auto        inn_shapes = e.inn_shapes();
    auto const& shape = inn_shapes.at(which_inn);
    return make_data(shape);
}

buffer_t make_out(einsummable_t const& e)
{
    return make_buffer(e.out_size());
}

void execute_direct(uint64_t batch_size, uint64_t start_pos, uint64_t dim, uint64_t n_heads)
{
    uint64_t head_dim = uint64_div(dim, n_heads, "head_dim");

    kernel_manager_t km;

    uint64_t unit = 1;

    auto es_a = make_e({batch_size, unit, n_heads, head_dim, n_heads, head_dim}, "abef,cdef->abcd");
    auto a0 = make_data(es_a, 0);
    auto a1 = make_data(es_a, 1);

    auto es_b = make_e({batch_size, n_heads, unit, start_pos, head_dim}, "acbe,adbe->abcd");
    auto b1 = make_data(es_b, 1);

    auto es_c = make_e({batch_size, n_heads, unit, head_dim, start_pos}, "abce,aebd->abcd");
    auto c1 = make_data(es_c, 1);

    auto es_d = make_e({batch_size, unit, n_heads, head_dim, n_heads, head_dim}, "aebf,cdef->abcd");
    auto d1 = make_data(es_d, 1);

    uint64_t wsz = 0;
    for (auto const& e : {es_a, es_b, es_c, es_d}) {
        wsz = std::max(wsz, km.build(e).value());
    }

    buffer_t                         workspace = make_buffer(wsz);
    optional<tuple<void*, uint64_t>> wspace = tuple<void*, uint64_t>(workspace->raw(), wsz);

    {
        gremlin_t timer("EXECUTE DIRECT");

        buffer_t b0 = make_out(es_a);
        km(es_a, b0->raw(), {a0->raw(), a1->raw()}, wspace);

        buffer_t c0 = make_out(es_b);
        km(es_b, c0->raw(), {b0->raw(), b1->raw()}, wspace);

        buffer_t d0 = make_out(es_c);
        km(es_c, d0->raw(), {c0->raw(), c1->raw()}, wspace);

        buffer_t e0 = make_out(es_d);
        km(es_d, e0->raw(), {d0->raw(), d1->raw()}, wspace);
    }
}

void execute_direct_nounit(uint64_t batch_size, uint64_t start_pos, uint64_t dim, uint64_t n_heads)
{
    uint64_t head_dim = uint64_div(dim, n_heads, "head_dim");

    kernel_manager_t km;

    uint64_t unit = 1;

    auto es_a = make_e({batch_size, n_heads, head_dim, n_heads, head_dim}, "ade,bcde->abc");
    auto a0 = make_data(es_a, 0);
    auto a1 = make_data(es_a, 1);

    auto es_b = make_e({batch_size, n_heads, start_pos, head_dim}, "abd,acbd->abc");
    auto b1 = make_data(es_b, 1);

    auto es_c = make_e({batch_size, n_heads, head_dim, start_pos}, "abd,adbc->abc");
    auto c1 = make_data(es_c, 1);

    auto es_d = make_e({batch_size, n_heads, head_dim, n_heads, head_dim}, "ade,bcde->abc");
    auto d1 = make_data(es_d, 1);

    uint64_t wsz = 0;
    for (auto const& e : {es_a, es_b, es_c, es_d}) {
        wsz = std::max(wsz, km.build(e).value());
    }

    buffer_t                         workspace = make_buffer(wsz);
    optional<tuple<void*, uint64_t>> wspace = tuple<void*, uint64_t>(workspace->raw(), wsz);

    {
        gremlin_t timer("AAAAA");

        buffer_t b0 = make_out(es_a);
        km(es_a, b0->raw(), {a0->raw(), a1->raw()}, wspace);

        buffer_t c0 = make_out(es_b);
        km(es_b, c0->raw(), {b0->raw(), b1->raw()}, wspace);

        buffer_t d0 = make_out(es_c);
        km(es_c, d0->raw(), {c0->raw(), c1->raw()}, wspace);

        buffer_t e0 = make_out(es_d);
        km(es_d, e0->raw(), {d0->raw(), d1->raw()}, wspace);
    }
}

void tblis_print(tblis::tblis_tensor& t)
{
    DOUT("t.ndim " << t.ndim);
    for (int i = 0; i != t.ndim; ++i) {
        DOUT("  len " << t.len[i]);
    }
    for (int i = 0; i != t.ndim; ++i) {
        DOUT("  str " << t.stride[i]);
    }

    // DOUT(dbuffer_t(dtype_t::f32, d));
    // for(int i = 0; i != product(shape_); ++i) {
    //   std::cout << "? " << reinterpret_cast<float*>(t.data)[i] << " ";
    // } std::cout << std::endl;
}

struct tblis_t {
    tblis_t(vector<uint64_t> const& shape_, bool rnd = false) : stride(shape_.size())
    {
        d = make_buffer(product(shape_) * dtype_size(dtype_t::f32));

        shape.reserve(shape_.size());
        for (auto const& s : shape_) {
            shape.push_back(s);
        }

        tblis::stride_type s = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride[i] = s;
            s *= shape[i];
        }

        tblis_init_tensor_s(
            &t, shape.size(), shape.data(), reinterpret_cast<float*>(d->raw()), stride.data());

        if (rnd) {
            random();
        } else {
            zeros();
        }
    }

    void random()
    {
        dbuffer_t dd(dtype_t::f32, d);
        dd.random("0.0", "1.0");
    }

    void zeros()
    {
        dbuffer_t dd(dtype_t::f32, d);
        dd.zeros();
    }

    void ones()
    {
        dbuffer_t dd(dtype_t::f32, d);
        dd.ones();
    }

    void print()
    {
        DOUT(dbuffer_t(dtype_t::f32, d));
    }

    tblis::tblis_tensor* ptr()
    {
        return &t;
    }
    tblis::tblis_tensor const* ptr() const
    {
        return &t;
    }

    vector<tblis::len_type>    shape;
    vector<tblis::stride_type> stride;
    tblis::tblis_tensor        t;
    buffer_t                   d;
};

void execute_tblis(uint64_t batch_size, uint64_t start_pos, uint64_t dim, uint64_t n_heads)
{
    uint64_t head_dim = uint64_div(dim, n_heads, "head_dim");
    uint64_t unit = 1;

    using tss_t = tuple<string, string>;
    vector<tss_t> ss_{
        tss_t("abef", "cdef"), tss_t("acbe", "adbe"), tss_t("abce", "aebd"), tss_t("aebf", "cdef")};

    vector<vector<uint64_t>> sz_0{{batch_size, unit, n_heads, head_dim},
                                  {batch_size, unit, n_heads, head_dim},
                                  {batch_size, n_heads, unit, start_pos},
                                  {batch_size, n_heads, unit, head_dim}};

    vector<vector<uint64_t>> sz_1{{n_heads, head_dim, n_heads, head_dim},
                                  {batch_size, start_pos, n_heads, head_dim},
                                  {batch_size, start_pos, n_heads, head_dim},
                                  {n_heads, head_dim, n_heads, head_dim}};

    tblis_t a0(sz_0[0], true);

    tblis_t a1(sz_1[0], true);
    tblis_t b1(sz_1[1], true);
    tblis_t c1(sz_1[2], true);
    tblis_t d1(sz_1[3], true);

    string abcd = "abcd";

    {
        gremlin_t timer("EXECUTE TBLIS");

        tblis_t b0(sz_0[1]);
        auto const& [al, ar] = ss_[0];
        tblis::tblis_tensor_mult(
            tci_single, NULL, a0.ptr(), al.data(), a1.ptr(), ar.data(), b0.ptr(), abcd.data());

        tblis_t c0(sz_0[2]);
        auto const& [bl, br] = ss_[1];
        tblis::tblis_tensor_mult(
            tci_single, NULL, b0.ptr(), bl.data(), b1.ptr(), br.data(), c0.ptr(), abcd.data());

        tblis_t d0(sz_0[3]);
        auto const& [cl, cr] = ss_[2];
        tblis::tblis_tensor_mult(
            tci_single, NULL, c0.ptr(), cl.data(), c1.ptr(), cr.data(), d0.ptr(), abcd.data());
    }
}

void tblis_mm(uint64_t ni, uint64_t nj, uint64_t nk)
{
    tblis_t lhs({ni, nj}, true);
    tblis_t rhs({nj, nk}, true);

    lhs.random();
    rhs.ones();

    tblis_t out({ni, nk});
    out.zeros();

    string s_lhs = "ij";
    string s_rhs = "jk";
    string s_out = "ik";

    lhs.print();
    rhs.print();
    out.print();

    tblis::tblis_tensor_mult(
        NULL, NULL, lhs.ptr(), s_lhs.data(), rhs.ptr(), s_rhs.data(), out.ptr(), s_out.data());

    out.print();
}

void tblis_add(uint64_t ni, uint64_t nj)
{
    tblis_t lhs({ni, nj}, true);
    tblis_t rhs({ni, nj, nj}, true);

    lhs.ones();
    rhs.zeros();

    lhs.print();
    rhs.print();

    string sl = "ij";
    string sr = "ijk";
    tblis::tblis_tensor_add(NULL, NULL, lhs.ptr(), sl.data(), rhs.ptr(), sr.data());

    rhs.print();
}

void tblis_exp01()
{
    using namespace tblis;
    DLINE;

    tblis_t aa({10, 9, 2, 5}, true);
    DOUT("AAAAAAAAAAAAAA");
    tblis_print(aa.t);
    // DLINEOUT(dbuffer_t(dtype_t::f32, aa.d));

    buffer_t ddd = make_buffer(10 * 9 * 2 * 5 * dtype_size(dtype_t::f32));

    float               data_A[10 * 9 * 2 * 5];
    tblis_tensor        A;
    vector<len_type>    l0{10, 9, 2, 5};
    vector<stride_type> s0{90, 10, 5, 1}; // s0{1, 10, 90, 180};
    // tblis_init_tensor_scaled_s(&A, 2.0f, 4, l0.data(),
    //                            data_A, s0.data());
    // tblis_init_tensor_s(&A, 4, l0.data(), data_A, s0.data());
    tblis_init_tensor_s(&A, 4, l0.data(), reinterpret_cast<float*>(ddd->raw()), s0.data());
    DLINE;

    // initialize data_A...

    float               data_B[5 * 9 * 2];
    tblis_tensor        B;
    vector<len_type>    l1{5, 9, 2};
    vector<stride_type> s1{18, 2, 1}; // s1{1,5,45};
    tblis_init_tensor_scaled_s(&B, 0.0f, 3, l1.data(), data_B, s1.data());
    DLINE;

    string asd = "ijkl";
    string dsa = "ljk";
    // this overwrites B with a scaled permuted trace of A
    tblis_tensor_add(NULL, NULL, &A, asd.data(), &B, dsa.data());
    DLINE;
    tblis_print(aa.t);
    tblis_tensor_add(NULL, NULL, aa.ptr(), asd.data(), &B, dsa.data());
    tblis_print(aa.t);
    DLINE;
    tblis_t bb({5, 9, 2});
    DLINE;
    // DLINEOUT(dbuffer_t(dtype_t::f32, aa.d));
    // DLINEOUT(dbuffer_t(dtype_t::f32, bb.d));
    tblis_tensor_add(NULL, NULL, aa.ptr(), asd.data(), bb.ptr(), dsa.data());
    // DLINEOUT(dbuffer_t(dtype_t::f32, bb.d));
    DLINE;
}

int main()
{
    uint64_t batch_size = 16;
    uint64_t start_pos = 1170;
    uint64_t dim = 4096;
    uint64_t n_heads = 32;

    // graph_t g = make_attention_graph(batch_size, start_pos, dim, n_heads);

    // std::ofstream f("g.gv");
    // g.print_graphviz(f);
    // DOUT("printed g.gv");

    // for(uint64_t sp = 1; sp != 1029; ++sp) {
    //   graph_t g = make_attention_graph(batch_size, sp, dim, n_heads);
    //   uint64_t ret = 0;
    //   for(auto const& node: g.nodes) {
    //     if(node.op.is_einsummable()) {
    //       ret += product(node.op.get_einsummable().join_shape);
    //     }
    //   }
    //   DOUT(ret);
    // }

    execute_tblis(batch_size, start_pos, dim, n_heads);
    execute_tblis(batch_size, start_pos, dim, n_heads);
    execute_direct(batch_size, start_pos, dim, n_heads);
    execute_direct(batch_size, start_pos, dim, n_heads);

    // for(uint64_t sp = 1; sp != 400; ++sp) {
    //   execute_tblis(batch_size, sp, dim, n_heads);
    // }

    // tblis_exp01();
    // DLINEOUT("???????????????????????????????????");

    // tblis_mm(3,4,5);
    // tblis_add(4,4);

    // execute_direct_nounit(batch_size, start_pos, dim, n_heads);
    // for(uint64_t sp = 1; sp != 1029; ++sp) {
    //   execute_direct_nounit(batch_size, sp, dim, n_heads);
    // }

    // execute_direct(batch_size, start_pos, dim, n_heads);

    // for(uint64_t sp = 1; sp != 1029; ++sp) {
    //   execute_direct(batch_size, sp, dim, n_heads);
    // }
}
