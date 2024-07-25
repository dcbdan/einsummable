template <typename T>
void permute_t::operator()(vector<uint64_t> dims, vector<int> perm, T* out, T const* inn) const
{
    // Some extra tensor-permute optimizations:
    // 1. fuse adjacent dimensions...
    //      so if perm is [2,0,1], fuse [0,1] yielding [1,0]
    // 2. remove dimensions of size 1
    while (dims.size() > 1 && (has_fuse(dims, perm) || has_singleton(dims, perm))) {
    }
    // dims and perm have been modified accordingly

    // This is a "batched" permutation if
    // the first indices are unpermuted... That is,
    //   perm = {1,0,2}   has a batch size of 1,
    //   perm = {0,2,1}   has a batch size of dims[0]
    //   perm = {0,1,3,2} has a batch size of dims[0]*dims[1]
    int      num_batch_dims = 0;
    uint64_t batch_size = 1;
    for (int i = 0; i != perm.size(); ++i) {
        if (perm[i] == i) {
            num_batch_dims += 1;
            batch_size *= dims[i];
        } else {
            break;
        }
    }

    // In this case, there is no permutation
    // and so it is just a copy.
    // For example,
    //   perm might equal {0,1,2,3,4,5}
    //   (if adjacent dimensions weren't fused)
    if (num_batch_dims == perm.size()) {
        std::copy(inn, inn + batch_size, out);
        return;
    }

    // In this case, there are no batch dimensions.
    // (This would be correct even if there were
    //  batch dimensions.)
    if (num_batch_dims == 0) {
        auto const [str_inn, str_out] = build_strides(dims, perm);

        vector<tuple<uint64_t, uint64_t>> rngs;
        rngs.reserve(dims.size());
        for (auto const& n : dims) {
            rngs.emplace_back(0, n);
        }
        recurse(rngs, str_inn, str_out, out, inn);
        return;
    }

    // This is a batched permutation; do each batch separately.
    // The idea being that doing this in batches will increase cache hits the most.

    vector<uint64_t> batch_dims;
    batch_dims.reserve(num_batch_dims);

    vector<int> batch_perm;
    batch_perm.reserve(num_batch_dims);

    int num_after_batch = dims.size() - num_batch_dims;
    for (int i = 0; i != num_after_batch; ++i) {
        int const& dim_size = dims[num_batch_dims + i];
        int const& which_dim_with_batch = perm[num_batch_dims + i];
        batch_dims.push_back(dim_size);
        batch_perm.push_back(which_dim_with_batch - num_batch_dims);
    }

    vector<tuple<uint64_t, uint64_t>> batch_rngs;
    uint64_t                          offset = 1;
    batch_rngs.reserve(batch_dims.size());
    for (auto const& n : batch_dims) {
        batch_rngs.emplace_back(0, n);
        offset *= n;
    }

    auto const& [str_inn, str_out] = build_strides(batch_dims, batch_perm);

    for (int which_batch = 0; which_batch != batch_size; ++which_batch) {
        recurse(batch_rngs, str_inn, str_out, out, inn);
        inn += offset;
        out += offset;
    }
}

template <typename T>
void permute_t::recurse(vector<tuple<uint64_t, uint64_t>>& rngs,
                        vector<uint64_t> const&            str_inn,
                        vector<uint64_t> const&            str_out,
                        T*                                 out,
                        T const*                           inn) const
{
    // Traverse over rngs to determine two things:
    //
    // 1. What is the block size being written to?
    //     > use this to see if this is the base case
    //
    // 2. Which rank has the largest remaining dimension?
    //     > if not the base case, recurse on this rank

    int block_size = 1;
    int which_recurse = 0;
    int largest_remaining = 0;
    for (int i = 0; i != rngs.size(); ++i) {
        auto const& [beg, end] = rngs[i];
        int remaining = end - beg;
        block_size *= remaining;

        if (remaining > largest_remaining) {
            largest_remaining = remaining;
            which_recurse = i;
        }
    }

    if (block_size < min_block_size) {
        // Here, directly dispatch the four loops based off of how many
        // dimensions there are.
        //
        // Doing for loops is way faster than using indexer.
        //
        if (rngs.size() == 2) {
            auto const& so0 = str_out[0];
            auto const& so1 = str_out[1];

            auto const& si0 = str_inn[0];
            auto const& si1 = str_inn[1];

            auto const& [b0, e0] = rngs[0];
            auto const& [b1, e1] = rngs[1];
            for (int i0 = b0; i0 != e0; ++i0) {
                for (int i1 = b1; i1 != e1; ++i1) {
                    out[i0 * so0 + i1 * so1] = inn[i0 * si0 + i1 * si1];
                }
            }
        } else if (rngs.size() == 3) {
            auto const& so0 = str_out[0];
            auto const& so1 = str_out[1];
            auto const& so2 = str_out[2];

            auto const& si0 = str_inn[0];
            auto const& si1 = str_inn[1];
            auto const& si2 = str_inn[2];

            auto const& [b0, e0] = rngs[0];
            auto const& [b1, e1] = rngs[1];
            auto const& [b2, e2] = rngs[2];

            for (int i0 = b0; i0 != e0; ++i0) {
                for (int i1 = b1; i1 != e1; ++i1) {
                    for (int i2 = b2; i2 != e2; ++i2) {
                        out[i0 * so0 + i1 * so1 + i2 * so2] = inn[i0 * si0 + i1 * si1 + i2 * si2];
                    }
                }
            }
        } else if (rngs.size() == 4) {
            auto const& so0 = str_out[0];
            auto const& so1 = str_out[1];
            auto const& so2 = str_out[2];
            auto const& so3 = str_out[3];

            auto const& si0 = str_inn[0];
            auto const& si1 = str_inn[1];
            auto const& si2 = str_inn[2];
            auto const& si3 = str_inn[3];

            auto const& [b0, e0] = rngs[0];
            auto const& [b1, e1] = rngs[1];
            auto const& [b2, e2] = rngs[2];
            auto const& [b3, e3] = rngs[3];

            for (int i0 = b0; i0 != e0; ++i0) {
                for (int i1 = b1; i1 != e1; ++i1) {
                    for (int i2 = b2; i2 != e2; ++i2) {
                        for (int i3 = b3; i3 != e3; ++i3) {
                            out[i0 * so0 + i1 * so1 + i2 * so2 + i3 * so3] =
                                inn[i0 * si0 + i1 * si1 + i2 * si2 + i3 * si3];
                        }
                    }
                }
            }
        } else if (rngs.size() == 5) {
            auto const& so0 = str_out[0];
            auto const& so1 = str_out[1];
            auto const& so2 = str_out[2];
            auto const& so3 = str_out[3];
            auto const& so4 = str_out[4];

            auto const& si0 = str_inn[0];
            auto const& si1 = str_inn[1];
            auto const& si2 = str_inn[2];
            auto const& si3 = str_inn[3];
            auto const& si4 = str_inn[4];

            auto const& [b0, e0] = rngs[0];
            auto const& [b1, e1] = rngs[1];
            auto const& [b2, e2] = rngs[2];
            auto const& [b3, e3] = rngs[3];
            auto const& [b4, e4] = rngs[4];

            for (int i0 = b0; i0 != e0; ++i0) {
                for (int i1 = b1; i1 != e1; ++i1) {
                    for (int i2 = b2; i2 != e2; ++i2) {
                        for (int i3 = b3; i3 != e3; ++i3) {
                            for (int i4 = b4; i4 != e4; ++i4) {
                                out[i0 * so0 + i1 * so1 + i2 * so2 + i3 * so3 + i4 * so4] =
                                    inn[i0 * si0 + i1 * si1 + i2 * si2 + i3 * si3 + i4 * si4];
                            }
                        }
                    }
                }
            }
        } else {
            throw std::runtime_error(
                "tensor permutation with more than five ranks not implemented!");
        }

        return;
    }

    auto [beg, end] = rngs[which_recurse];
    uint64_t half = beg + ((end - beg) / 2);

    rngs[which_recurse] = {beg, half};
    recurse(rngs, str_inn, str_out, out, inn);

    rngs[which_recurse] = {half, end};
    recurse(rngs, str_inn, str_out, out, inn);

    // Note: rngs is passed by reference, so on the recursion out,
    //       we have set rngs back to the way it was
    // (Bassing by copy is less to think about, but not as efficient)
    rngs[which_recurse] = {beg, end};
}
