#include "aggplan.h"

set<int> agg_plan_t::source_locs() const
{
    set<int> ret;
    for (int l = bloc; l != eloc; ++l) {
        ret.insert(l);
    }
    return ret;
}

int agg_plan_t::loc_at(int i) const
{
    i = i % (eloc - bloc);
    return bloc + i;
}

vector<int> agg_plan_t::as_vector(int nagg) const
{
    vector<int> ret;
    ret.reserve(nagg);
    for (int i = 0; i != nagg; ++i) {
        ret.push_back(loc_at(i));
    }
    return ret;
}

// I'm sure there is a faster way with bit manipulation,
// but computers are fast, as it turns out
bool _is_power_of_2(int x)
{
    if (x <= 0) {
        return false;
    }
    while (x % 2 == 0) {
        x /= 2;
    }
    return x == 1;
}

vector<agg_plan_t> gen_agg_plans(int nloc, int nagg)
{
    if (!_is_power_of_2(nloc)) {
        throw std::runtime_error("nloc must be a power of 2");
    }
    if (!_is_power_of_2(nagg)) {
        throw std::runtime_error("nagg must be a power of 2");
    }

    vector<agg_plan_t> ret;
    ret.reserve(nloc * 2);
    for (int sz = std::min(nloc, nagg); sz != 0; sz /= 2) {
        for (int begl = 0; begl != nloc; begl += sz) {
            ret.emplace_back(begl, begl + sz);
        }
    }

    return ret;
}
