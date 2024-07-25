#include "select.h"

#include "../base/partition.h" // partitions_region

select_t::select_t(dtype_t                               dtype,
                   vector<uint64_t> const&               os,
                   vector<select_t::inn_region_t> const& irs)
    : dtype(dtype), out_shape(os), inn_regions(irs)
{
    int rank = out_shape.size();
    if (rank == 0) {
        throw std::runtime_error("cannot have empty out shape in select");
    }
    for (auto const& d : out_shape) {
        if (d == 0) {
            throw std::runtime_error("cannot have dimension of size zero");
        }
    }

    vector<vector<tuple<uint64_t, uint64_t>>> hrects;
    hrects.reserve(inn_regions.size());
    for (auto const& inn_region : inn_regions) {
        if (inn_region.size() != rank) {
            throw std::runtime_error("inn region must have same rank as output");
        }

        hrects.emplace_back();
        auto& hrect = hrects.back();
        hrect.reserve(rank);

        for (int i = 0; i != rank; ++i) {
            auto const& d_out = out_shape[i];
            auto const& [d_inn, offset_inn, offset_out, size] = inn_region[i];

            if (size + offset_out > d_out || size + offset_inn > d_inn || d_inn == 0 || size == 0) {
                throw std::runtime_error("invalid select given");
            }

            hrect.emplace_back(offset_out, offset_out + size);
        }
    }

    if (!partitions_region(hrects, out_shape)) {
        throw std::runtime_error("This select does not partition the write region");
    }
}

vector<touch_t> select_t::as_touches() const
{
    vector<touch_t> ret;
    ret.reserve(inn_regions.size());
    for (int i = 0; i != inn_regions.size(); ++i) {
        ret.push_back(as_touch(i));
    }
    return ret;
}

touch_t select_t::as_touch(int which) const
{
    vector<touchdim_t> tds;
    auto const&        sds = inn_regions.at(which);
    for (int i = 0; i != sds.size(); ++i) {
        uint64_t    d_out = out_shape[i];
        auto const& sd = sds[i];
        tds.push_back(touchdim_t{.d_inn = sd.d_inn,
                                 .d_out = d_out,
                                 .offset_inn = sd.offset_inn,
                                 .offset_out = sd.offset_out,
                                 .size = sd.size});
    }

    return touch_t{.selection = tds, .castable = std::nullopt, .dtype = dtype};
}

vector<uint64_t> select_t::wrt_output_point(vector<uint64_t> const& inn_point, int which_inn) const
{
    auto const& inn_region = inn_regions.at(which_inn);
    int         rank = inn_region.size();

    vector<uint64_t> out_point;
    out_point.reserve(rank);
    for (int i = 0; i != rank; ++i) {
        auto const&     sd = inn_region[i];
        uint64_t const& b_inn = inn_point[i];
        out_point.push_back(sd.offset_out + (b_inn - sd.offset_inn));
    }

    return out_point;
}

hrect_t select_t::wrt_output_hrect(hrect_t const& inn_hrect, int which_inn) const
{
    auto ret_b = wrt_output_point(vector_mapfst(inn_hrect), which_inn);
    auto ret_e = wrt_output_point(vector_mapsnd(inn_hrect), which_inn);
    return vector_zip(ret_b, ret_e);
}

hrect_t select_t::wrt_output_inn_hrect(int which_input) const
{
    hrect_t ret;
    for (auto const& sd : inn_regions.at(which_input)) {
        ret.emplace_back(sd.offset_out, sd.offset_out + sd.size);
    }
    return ret;
}

hrect_t select_t::wrt_input_inn_hrect(int which_input) const
{
    hrect_t ret;
    for (auto const& sd : inn_regions.at(which_input)) {
        ret.emplace_back(sd.offset_inn, sd.offset_inn + sd.size);
    }
    return ret;
}

vector<tuple<hrect_t, int>> select_t::collect(hrect_t out_hrect) const
{
    // for each input, does it intersect, if so, where?
    vector<tuple<hrect_t, int>> ret;
    int                         rank = out_shape.size();
    for (int which_inn = 0; which_inn != inn_regions.size(); ++which_inn) {
        auto const&                       inn_region = inn_regions[which_inn];
        vector<tuple<uint64_t, uint64_t>> inn_hrect;
        for (int i = 0; i != rank; ++i) {
            selectdim_t const& sd = inn_region[i];
            auto const& [out_blk_b, out_blk_e] = out_hrect[i];
            uint64_t const& here_b = sd.offset_out;
            uint64_t        here_e = sd.offset_out + sd.size;
            uint64_t        out_b = std::max(out_blk_b, here_b);
            uint64_t        out_e = std::min(out_blk_e, here_e);
            if (out_b < out_e) {
                inn_hrect.emplace_back(sd.offset_inn + (out_b - sd.offset_out),
                                       sd.offset_inn + (out_e - sd.offset_out));
            } else {
                break;
            }
        }
        if (inn_hrect.size() == rank) {
            ret.emplace_back(inn_hrect, which_inn);
        }
    }

    return ret;
}

vector<uint64_t> select_t::inn_shape(int which_inn) const
{
    return vector_from_each_member(inn_regions[which_inn], uint64_t, d_inn);
}

select_t make_concat(int dim, dtype_t dtype, vector<vector<uint64_t>> const& input_shapes)
{
    using selectdim_t = select_t::selectdim_t;

    if (input_shapes.size() == 0) {
        throw std::runtime_error("cannot concat empty list");
    }

    uint64_t offset_dim = 0;

    vector<vector<selectdim_t>> sdss;
    sdss.reserve(input_shapes.size());

    for (auto const& inn_shape : input_shapes) {
        sdss.emplace_back();
        auto& sds = sdss.back();
        for (int i = 0; i != inn_shape.size(); ++i) {
            uint64_t const& d_inn = inn_shape[i];
            if (i == dim) {
                sds.push_back(selectdim_t{
                    .d_inn = d_inn, .offset_inn = 0, .offset_out = offset_dim, .size = d_inn});
                offset_dim += d_inn;
            } else {
                sds.push_back(
                    selectdim_t{.d_inn = d_inn, .offset_inn = 0, .offset_out = 0, .size = d_inn});
            }
        }
    }
    vector<uint64_t> out_shape = vector_from_each_member(sdss[0], uint64_t, size);
    out_shape[dim] = offset_dim;

    return select_t(dtype, out_shape, sdss);
}

select_t make_subset(dtype_t                                  dtype,
                     vector<tuple<uint64_t, uint64_t>> const& hrect,
                     vector<uint64_t>                         inn_shape)
{
    using selectdim_t = select_t::selectdim_t;

    int rank = hrect.size();
    if (inn_shape.size() != rank || rank == 0) {
        throw std::runtime_error("invalid input to make_subset");
    }

    vector<selectdim_t> sds;
    vector<uint64_t>    out_shape;
    sds.reserve(rank);
    out_shape.reserve(rank);
    for (int i = 0; i != rank; ++i) {
        auto const& [beg, end] = hrect[i];
        uint64_t size = end - beg;
        sds.push_back(
            selectdim_t{.d_inn = inn_shape[i], .offset_inn = beg, .offset_out = 0, .size = size});
        out_shape.push_back(size);
    }

    return select_t(dtype, out_shape, {sds});
}
