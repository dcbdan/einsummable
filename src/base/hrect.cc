#include "hrect.h"

vector<tuple<uint64_t, uint64_t>> hrect_center(vector<tuple<uint64_t, uint64_t>> const& full,
                                               vector<tuple<uint64_t, uint64_t>> const& small)
{
    if (full.size() != small.size()) {
        throw std::runtime_error("hrect_center: incorrect sizes");
    }

    vector<tuple<uint64_t, uint64_t>> ret;
    ret.reserve(full.size());

    for (int i = 0; i != full.size(); ++i) {
        auto const& [fb, fe] = full[i];
        auto const& [sb, se] = small[i];
        if (fb <= sb && sb < se && se <= fe) {
            ret.emplace_back(sb - fb, se - fb);
        } else {
            throw std::runtime_error("hrect_center: invalid");
        }
    }

    return ret;
}

vector<uint64_t> hrect_shape(vector<tuple<uint64_t, uint64_t>> const& hrect)
{
    vector<uint64_t> ret;
    ret.reserve(hrect.size());
    for (auto const& [x, y] : hrect) {
        ret.push_back(y - x);
    }
    return ret;
}

vector<tuple<uint64_t, uint64_t>> hrect_intersect(vector<tuple<uint64_t, uint64_t>> const& lhs,
                                                  vector<tuple<uint64_t, uint64_t>> const& rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("hrect_intersect: incorrect sizes");
    }

    vector<tuple<uint64_t, uint64_t>> ret;
    ret.reserve(lhs.size());
    for (int i = 0; i != lhs.size(); ++i) {
        auto const& [lb, le] = lhs[i];
        auto const& [rb, re] = rhs[i];

        uint64_t b = std::max(lb, rb);
        uint64_t e = std::min(le, re);

        if (b >= e) {
            throw std::runtime_error("empty intersection");
        }
        ret.emplace_back(b, e);
    }

    return ret;
}

optional<tuple<uint64_t, uint64_t>> interval_intersect(tuple<uint64_t, uint64_t> const& lhs,
                                                       tuple<uint64_t, uint64_t> const& rhs)
{
    auto const& [lb, le] = lhs;
    auto const& [rb, re] = rhs;

    uint64_t b = std::max(lb, rb);
    uint64_t e = std::min(le, re);

    if (b < e) {
        return tuple<uint64_t, uint64_t>{b, e};
    } else {
        return std::nullopt;
    }
}

vector<tuple<uint64_t, uint64_t>> hrect_full_hrect_from_shape(vector<uint64_t> const& xs)
{
    return vector_zip(vector<uint64_t>(xs.size(), 0), xs);
}
