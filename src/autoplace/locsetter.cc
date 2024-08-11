#include "locsetter.h"

loc_setter_t::loc_setter_t(int n_items, int n_locs)
{
    avail_locs = vector<int>(n_locs);
    std::iota(avail_locs.begin(), avail_locs.end(), 0);

    int cnt = (n_items / n_locs) + 1;
    remainder = n_items % n_locs;
    num_remaining = vector<int>(n_locs, cnt);
}

bool loc_setter_t::decrement(int loc)
{
    auto al_iter = std::find(avail_locs.begin(), avail_locs.end(), loc);
    if (al_iter == avail_locs.end()) {
        throw std::runtime_error("invalid decrement: this loc isn't here");
    }
    int which = al_iter - avail_locs.begin();

    auto nr_iter = num_remaining.begin() + which;

    int& nr = *nr_iter;
    nr -= 1;

    if (nr == 0 && remainder == 0) {
        throw std::runtime_error("should not happen: loc setter");
    } else if (nr == 0 && remainder > 0) {
        remainder -= 1;
        num_remaining.erase(nr_iter);
        avail_locs.erase(al_iter);
        if (remainder == 0) {
            // remove all locs with nr == 1 now that the remainder
            // has run out
            for (int i = avail_locs.size() - 1; i >= 0; --i) {
                if (num_remaining[i] == 1) {
                    num_remaining.erase(num_remaining.begin() + i);
                    avail_locs.erase(avail_locs.begin() + i);
                }
            }
        }
        return false;
    } else if (nr == 1 && remainder == 0) {
        num_remaining.erase(nr_iter);
        avail_locs.erase(al_iter);
        return false;
    } else {
        return true;
    }
}

bool loc_setter_t::is_avail(int loc) const
{
    auto iter = std::find(avail_locs.begin(), avail_locs.end(), loc);
    return iter != avail_locs.end();
};

int loc_setter_t::runif() const
{
    int which_loc = ::runif(avail_locs.size());
    return avail_locs[which_loc];
}

bool is_balanced(vector<int> const& locs)
{
    vector<int> cnts;
    for (auto const& loc : locs) {
        if (cnts.size() <= loc) {
            cnts.resize(loc + 1);
        }
        cnts[loc] += 1;
    }
    int const& mx = *(std::max_element(cnts.begin(), cnts.end()));
    int const& mn = *(std::min_element(cnts.begin(), cnts.end()));
    return mx - mn <= 1;
}
