#include "../src/execution/cpu/permute.h"

#include "../src/einsummable/reference.h"

vector<int> inverse_permute(vector<int> const& ps)
{
    map<int, int> m;
    for (int i = 0; i != ps.size(); ++i) {
        m.insert({ps[i], i});
    }
    vector<int> ret;
    for (int i = 0; i != ps.size(); ++i) {
        ret.push_back(m.at(i));
    }
    return ret;
}

template <typename T>
vector<T> permute_vector(vector<int> const& ps, vector<T> const& xs)
{
    vector<T> ret;
    ret.reserve(xs.size());
    for (auto const& p : ps) {
        ret.push_back(xs[p]);
    }
    return ret;
}

int main()
{
    // DOUT(inverse_permute({0,1,2})); // 012->012
    // DOUT(inverse_permute({2,1,0})); // from 210->012 to 012->210
    // DOUT(inverse_permute({0,2,1})); // from 021->012 to 012->021
    // DOUT(inverse_permute({1,2,0})); // from 120->012 to 012->201

    vector<uint64_t> inn_shape{10, 11, 12, 130};
    uint64_t         sz = product(inn_shape);

    auto make_einsummable = [&](vector<int> const& out_permute) {
        return einsummable_t(permute_vector(out_permute, inn_shape),
                             {inverse_permute(out_permute)},
                             int(inn_shape.size()),
                             scalarop_t::make_identity());
    };

    vector<int> permute(inn_shape.size());
    std::iota(permute.begin(), permute.end(), 0);

    dtype_t   dtype = dtype_t::f32;
    dbuffer_t inn = make_dbuffer(dtype, sz);
    inn.iota();

    dbuffer_t out1 = make_dbuffer(dtype, sz);
    dbuffer_t out2 = make_dbuffer(dtype, sz);

    out1.zeros();
    out2.zeros();

    permute_t kernel(1024);
    do {
        {
            auto                      einsummable = make_einsummable(permute);
            raii_print_time_elapsed_t gremlin("reference   ");
            reference_einsummable_inplace(einsummable, out1, {inn});
        }

        {
            raii_print_time_elapsed_t gremlin("kernel      ");
            kernel(inn_shape, permute, out2.f32(), inn.f32());
        }

        if (out1 != out2) {
            DOUT("reference " << out1);
            DOUT("kernel    " << out2);
            throw std::runtime_error("incorrect permutation!");
        }
    } while (std::next_permutation(permute.begin(), permute.end()));
}
