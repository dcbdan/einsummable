#pragma once
#include "../base/setup.h"

// A load balanced setter
struct loc_setter_t {
    loc_setter_t(int n_items, int n_locs);

    // return whether or not this loc is still available
    bool decrement(int loc);

    bool is_avail(int loc) const;

    vector<int> const& get_avail_locs() const
    {
        return avail_locs;
    }

    int runif() const;

private:
    vector<int> num_remaining;
    int         remainder;
    vector<int> avail_locs;
};

bool is_balanced(vector<int> const& locs);
