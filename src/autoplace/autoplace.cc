#include "autoplace.h"

#include "apart.h"
#include "alocate.h"

vector<placement_t> autoplace01(
  graph_t const& graph,
  autoplace_config_t const& config)
{
  vector<partition_t> parts = apart01(
    graph,
    config.n_compute(),
    config.max_branching(),
    config.discount_input_factor(),
    config.search_space());

  return alocate01(
    graph,
    parts,
    config.n_locs(),
    config.flops_per_byte_moved());
}

vector<placement_t> autoplace02(
  graph_t const& graph,
  autoplace_config_t const& config,
  map<int, placement_t> const& fixed_pls,
  vector<tuple<int,int>> const& equal_pls)
{
  map<int, partition_t> fixed_parts;
  map<int, vtensor_t<int>> fixed_locs;
  for(auto const& [id, pl]: fixed_pls) {
    fixed_parts.insert({id, pl.partition});
    fixed_locs.insert({id, pl.locations});
  }

  vector<partition_t> parts = apart02(
    graph,
    config.n_compute(),
    fixed_parts,
    equal_pls);

  auto ret = alocate02(
    graph, parts,
    config.n_locs(), config.flops_per_byte_moved(),
    fixed_locs,
    equal_pls);

  return ret;
}

equal_holder_t::equal_holder_t(vector<tuple<int, int>> const& eqs)
{
  for(auto const& [i,j]: eqs) {
    insert(i,j);
  }
}

bool equal_holder_t::has(int i) const {
  return lookup.count(i) > 0;
}

set<int> const& equal_holder_t::operator[](int i) const
{
  return sets.at(lookup.at(i));
}

void equal_holder_t::insert(int i, int j) {
  auto iter_i = lookup.find(i);
  auto iter_j = lookup.find(j);

  bool has_i = iter_i != lookup.end();
  bool has_j = iter_j != lookup.end();
  if(has_i && has_j) {
    int which_set_i = iter_i->second;
    int which_set_j = iter_j->second;
    if(which_set_i == which_set_j) {
      // nothing to do since they are both already
      // in the same set
    } else {
      // set_i and set_j need to be merged
      {
        set<int>& s = sets[which_set_i];
        set<int> const& other = sets[which_set_j];
        s.insert(other.begin(), other.end());
      }

      // get rid of set j,
      sets.erase(sets.begin() + which_set_j);

      // Fix the invalidated ints..
      // For each which_set value:
      //   if which_set_j, remap to to which_set_i
      //   else if which_set > which_set_j, remap to one less
      for(auto& [k, which_set_k]: lookup) {
        if(which_set_k == which_set_j) {
          which_set_k = which_set_i;
        } else if(which_set_k > which_set_j) {
          which_set_k--;
        }
      }
    }
  } else if(has_i) {
    int which_set = iter_i->second;
    lookup.insert({j, which_set});
    sets[which_set].insert(j);
  } else if(has_j) {
    int which_set = iter_j->second;
    lookup.insert({i, which_set});
    sets[which_set].insert(i);
  } else {
    sets.push_back(set<int>{i,j});
    lookup.insert({i, sets.size() - 1});
    lookup.insert({j, sets.size() - 1});
  }
}

std::ostream& operator<<(std::ostream& out, equal_holder_t const& e) {
  for(auto const& s: e.sets) {
    out << vector<int>(s.begin(), s.end());
  }
  return out;
}


