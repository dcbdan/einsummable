#include "touch.h"

touch_t touch_t::simplify() const
{
    vector<touchdim_t> new_selection;
    new_selection.push_back(selection[0]);

    auto is_dummy_dim = [](touchdim_t const& td) {
        auto const& [d_inn, d_out, o_inn, o_out, sz] = td;
        return d_inn == d_out && o_inn == 0 && o_out == 0 && d_inn == sz;
    };

    for (int i = 1; i != selection.size(); ++i) {
        if (is_dummy_dim(selection[i])) {
            int const& d = selection[i].d_inn;
            auto& [d_inn, d_out, o_inn, o_out, sz] = new_selection.back();
            d_inn *= d;
            d_out *= d;
            o_inn *= d;
            o_out *= d;
            sz *= d;
        } else {
            new_selection.push_back(selection[i]);
        }
    }

    return touch_t{.selection = new_selection, .castable = castable, .dtype = dtype};
}

string touch_t::to_wire() const
{
    es_proto::Touch r;
    to_proto(r);
    string ret;
    r.SerializeToString(&ret);
    return ret;
}

void touch_t::to_proto(es_proto::Touch& r) const
{
    for (auto const& [d_inn, d_out, offset_inn, offset_out, size] : selection) {
        es_proto::TouchDim* td = r.add_selection();

        td->set_d_inn(d_inn);
        td->set_d_out(d_out);
        td->set_offset_inn(offset_inn);
        td->set_offset_out(offset_out);
        td->set_size(size);
    }

    if (castable) {
        r.set_castable(write_with_ss(castable.value()));
    }

    r.set_dtype(write_with_ss(dtype));
}

touch_t touch_t::from_wire(string const& str)
{
    es_proto::Touch r;
    if (!r.ParseFromString(str)) {
        throw std::runtime_error("could not parse relation!");
    }
    return from_proto(r);
}

touch_t touch_t::from_proto(es_proto::Touch const& r)
{
    optional<castable_t> castable = std::nullopt;
    if (r.has_castable()) {
        castable = parse_with_ss<castable_t>(r.castable());
    }

    vector<touchdim_t> selection;
    selection.reserve(r.selection_size());

    for (int i = 0; i != r.selection_size(); ++i) {
        auto const& td = r.selection(i);
        selection.push_back(touchdim_t{.d_inn = td.d_inn(),
                                       .d_out = td.d_out(),
                                       .offset_inn = td.offset_inn(),
                                       .offset_out = td.offset_out(),
                                       .size = td.size()});
    }

    return touch_t{
        .selection = selection,
        .castable = castable,
        .dtype = parse_with_ss<dtype_t>(r.dtype()),
    };
}

std::size_t touch_t::hash() const
{
    std::hash<int>      h_int;
    std::hash<uint64_t> h_uint;

    std::size_t ret = h_int(int(dtype));
    if (castable) {
        hash_combine_impl(ret, h_int(int(castable.value())));
    } else {
        hash_combine_impl(ret, h_int(-1));
    }

    for (auto const& [a, b, c, d, e] : selection) {
        hash_combine_impl(ret, a);
        hash_combine_impl(ret, b);
        hash_combine_impl(ret, c);
        hash_combine_impl(ret, d);
        hash_combine_impl(ret, e);
    }

    return ret;
}

vector<touchdim_t>
make_touch_selection_from_full_small(vector<tuple<uint64_t, uint64_t>> const& full,
                                     vector<tuple<uint64_t, uint64_t>> const& small)
{
    if (full.size() != small.size()) {
        throw std::runtime_error("mtsffs: incorrect sizes");
    }

    vector<touchdim_t> ret;
    ret.reserve(full.size());
    for (int i = 0; i != full.size(); ++i) {
        auto const& [fb, fe] = full[i];
        auto const& [sb, se] = small[i];
        if (fb <= sb && sb < se && se <= fe) {
            ret.push_back(touchdim_t{.d_inn = se - sb,
                                     .d_out = fe - fb,
                                     .offset_inn = 0,
                                     .offset_out = sb - fb,
                                     .size = se - sb});
        } else {
            throw std::runtime_error("mtsffs: invalid");
        }
    }

    return ret;
}

optional<touch_t> touch_compose(touch_t const& a, touch_t const& b)
{
    if (a.dtype != b.dtype) {
        throw std::runtime_error("cannot touch compose with diff dtypes");
    }
    if (a.castable) {
        throw std::runtime_error("cannot compose if f in f(g(x)) has castable");
    }

    {
        vector<uint64_t> a_out_shape = vector_from_each_member(a.selection, uint64_t, d_out);
        vector<uint64_t> b_inn_shape = vector_from_each_member(b.selection, uint64_t, d_inn);
        if (!vector_equal(a_out_shape, b_inn_shape)) {
            throw std::runtime_error("touch compose x -f> y -g> z: f out g inn shape mismatch");
        }
    }
    int rank = a.selection.size();

    vector<touchdim_t> ss;
    ss.reserve(rank);
    for (int i = 0; i != rank; ++i) {
        auto const& td_xy = a.selection[i];
        auto const& td_yz = b.selection[i];

        uint64_t const& ox = td_xy.offset_inn;
        uint64_t const& oy_xy = td_xy.offset_out;
        uint64_t const& sx = td_xy.size;

        uint64_t const& oy_yz = td_yz.offset_inn;
        uint64_t const& oz = td_yz.offset_out;
        uint64_t const& sz = td_yz.size;

        uint64_t gx, gy, gz;

        // Example:
        //
        //        |
        //  |.    |.
        //  |.    |.    |.    <- the dots are being touched
        //  |     |     |
        //  |     |     |
        //        |     |
        //
        //  x     y     z
        //
        //  gx = 1
        //  gy = 0
        //  gz = 2  <- these are at the top of each vector
        //
        //  bxz = 2 <- these are with respect to all starts shifted
        //  exz = 3    to the same spot
        //
        //  ox    = 0 <- these are the offsets for the two
        //  oy_xy = 1    transformations (xy) and (yz)
        //  oy_yz = 2
        //  oz    = 0
        //
        // gx + ox    = gy + oy_xy
        // gy + oy_yz = gz + oz
        //
        // Solve this set of equations such
        // that gx == 0 || gy == 0 || gz == 0
        // and gx >= 0, gy >= 0 and gz >= 0

        if (ox <= oy_xy) {
            if (oz <= oy_yz) {
                // oy_xy - ox
                // oy_yz - oz
                gy = 0;
                gx = oy_xy - ox;
                gz = oy_yz - oz;
            } else {
                // oy_xy - ox
                // oz - oy_yz
                gz = 0;
                gy = oz - oy_yz;
                gx = gy + (oy_xy - ox);
            }
        } else {
            if (oz <= oy_yz) {
                // ox - oy_xy
                // oy_yz - oz
                gx = 0;
                gy = ox - oy_xy;
                gz = gy + (oy_yz - oz);
            } else {
                // ox - oy_xy
                // oz - oy_yz
                uint64_t xx = ox - oy_xy;
                uint64_t zz = oz - oy_yz;
                if (xx <= zz) {
                    gz = 0;
                    gy = zz;
                    gx = zz - xx;
                } else {
                    gx = 0;
                    gy = xx;
                    gz = xx - zz;
                }
            }
        }

        uint64_t bx = gx + ox;
        uint64_t bz = gz + oz;

        uint64_t ex = bx + sx;
        uint64_t ez = bz + sz;

        uint64_t b_xz = std::max(bx, bz);
        uint64_t e_xz = std::min(ex, ez);

        if (e_xz <= b_xz) {
            // the areas do not have an overlap
            return std::nullopt;
        }

        ss.push_back(touchdim_t{.d_inn = td_xy.d_inn,
                                .d_out = td_yz.d_out,
                                .offset_inn = b_xz - gx,
                                .offset_out = b_xz - gz,
                                .size = e_xz - b_xz});
    }

    return optional<touch_t>(touch_t{.selection = ss, .castable = b.castable, .dtype = b.dtype});
}

bool operator==(touchdim_t const& lhs, touchdim_t const& rhs)
{
    auto const& [q, w, e, r, t] = lhs;
    auto const& [a, s, d, f, g] = rhs;
    return q == a && w == s && e == d && r == f && t == g;
}

bool operator!=(touchdim_t const& lhs, touchdim_t const& rhs)
{
    return !(lhs == rhs);
}

bool operator==(touch_t const& lhs, touch_t const& rhs)
{
    if (lhs.dtype != rhs.dtype) {
        return false;
    }

    if (lhs.castable && rhs.castable) {
        if (lhs.castable.value() != rhs.castable.value()) {
            return false;
        }
    } else if (!lhs.castable && !rhs.castable) {
        // good
    } else {
        return false;
    }

    return vector_equal(lhs.selection, rhs.selection);
}

bool operator!=(touch_t const& lhs, touch_t const& rhs)
{
    return !(lhs == rhs);
}
