#include "relation.h"

#include "einsummable.pb.h"

relation_t relation_t::make_singleton(
  dtype_t const& dtype,
  vector<uint64_t> const& shape,
  int id,
  int loc)
{
  vector<int> ones(shape.size(), 1);
  placement_t pl(partition_t::singleton(shape));
  pl.locations.get()[0] = loc;
  return relation_t {
    .dtype = dtype,
    .placement = pl,
    .tids = vtensor_t<int>(ones, {id})
  };
}

relation_t relation_t::as_singleton(int id, int loc) const {
  auto shape = placement.total_shape();
  return make_singleton(dtype, shape, id, loc);
}

string relation_t::to_wire() const {
  es_proto::Relation r;
  to_proto(r);
  string ret;
  r.SerializeToString(&ret);
  return ret;
}

void relation_t::to_proto(es_proto::Relation& r) const {
  r.set_dtype(write_with_ss(dtype));

  es_proto::Placement* p = r.mutable_placement();
  placement.to_proto(*p);

  for(auto const& t: tids.get()) {
    r.add_tids(t);
  }
}

relation_t relation_t::from_wire(string const& str) {
  es_proto::Relation r;
  if(!r.ParseFromString(str)) {
    throw std::runtime_error("could not parse relation!");
  }
  return from_proto(r);
}

relation_t relation_t::from_proto(es_proto::Relation const& r) {
  placement_t placement = placement_t::from_proto(r.placement());

  vector<int> tids;
  tids.reserve(r.tids_size());
  for(int i = 0; i != r.tids_size(); ++i) {
    tids.push_back(r.tids(i));
  }

  return relation_t {
    .dtype = parse_with_ss<dtype_t>(r.dtype()),
    .placement = placement,
    .tids = vtensor_t<int>(placement.block_shape(), tids)
  };
}

void remap_relations_t::insert(relation_t const& src, relation_t const& dst)
{
  if(src.dtype != dst.dtype) {
    throw std::runtime_error("remap insert: must have same dtype");
  }
  if(!vector_equal(
      src.placement.total_shape(),
      dst.placement.total_shape()))
  {
    throw std::runtime_error("remap insert: must have same total shape!");
  }

  remap.emplace_back(src, dst);
}

string remap_relations_t::to_wire() const {
  es_proto::RemapRelations r;
  for(auto const& [src,dst]: remap) {
    es_proto::Relation* s = r.add_srcs();
    src.to_proto(*s);

    es_proto::Relation* d = r.add_dsts();
    dst.to_proto(*d);
  }
  string ret;
  r.SerializeToString(&ret);
  return ret;
}

remap_relations_t
remap_relations_t::from_wire(string const& str) {
  es_proto::RemapRelations r;
  if(!r.ParseFromString(str)) {
    throw std::runtime_error("could not parse remap relations!");
  }

  if(r.srcs_size() != r.dsts_size()) {
    throw std::runtime_error("remap_relations must have same num srcs as dsts");
  }

  remap_relations_t ret;
  for(int i = 0; i != r.srcs_size(); ++i) {
    ret.insert(
      relation_t::from_proto(r.srcs(i)),
      relation_t::from_proto(r.dsts(i)));
  }

  return ret;
}

