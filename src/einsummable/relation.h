#pragma once
#include "../base/setup.h"

#include "../base/placement.h"

#include "scalarop.h"

struct relation_t {
  dtype_t dtype;
  placement_t placement;
  vtensor_t<int> tids;

  relation_t as_singleton(int id, int loc = 0) const;

  static relation_t from_wire(string const& str);
  static relation_t from_proto(es_proto::Relation const& r);

  string to_wire() const;
  void to_proto(es_proto::Relation& r) const;
};

struct remap_relations_t {
  void insert(relation_t const& src, relation_t const& dst);

  string to_wire() const;
  static remap_relations_t from_wire(string const& str);

  vector<tuple<relation_t, relation_t>> remap;
};
