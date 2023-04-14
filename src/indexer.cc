#include "indexer.h"

bool is_valid_index(vector<int> const& shape, int index) {
  return indexer_utils<int>::is_valid_index(shape, index);
}

bool is_valid_idxs(vector<int> const& shape, vector<int> const& idxs) {
  return indexer_utils<int>::is_valid_idxs(shape, idxs);
}

int idxs_to_index(vector<int> const& shape, vector<int> const& idxs) {
  return indexer_utils<int>::idxs_to_index(shape, idxs);
}

vector<int> index_to_idxs(vector<int> const& shape, int index) {
  return indexer_utils<int>::index_to_idxs(shape, index);
}

bool increment_idxs(vector<int> const& shape, vector<int>& idxs) {
  return indexer_utils<int>::increment_idxs(shape, idxs);
}

// (Mostly the same as increment_idxs)
bool increment_idxs_region(
  vector<tuple<int,int>> const& region,
  vector<int>& idxs)
{
  return indexer_utils<int>::increment_idxs_region(region, idxs);
}
