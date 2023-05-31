#include "cluster.h"

cluster_t::device_t::device_t(uint64_t c)
  : device_t(c, 1, [](einsummable_t const&){ return 1; })
{}

cluster_t::device_t::device_t(
  uint64_t cc,
  int cp,
  std::function<int(einsummable_t const&)> gu)
  : compute(cc), capacity(cp), get_util(gu)
{}

cluster_t cluster_t::make(
  vector<cluster_t::device_t> const& devices,
  vector<cluster_t::connection_t> const& cs)
{
  // Remove duplicate connections in cs
  // by summing the corresponding bandwidth

  vector<connection_t> connections;
  map<tuple<int, int>, int> to_connection;

  for(auto const& c: cs) {
    auto key = c.key();

    if(to_connection.count(key) > 0) {
      connections[to_connection[key]].bandwidth += c.bandwidth;
    } else {
      connections.push_back(c);
      to_connection.insert({key, connections.size()-1});
    }
  }

  return cluster_t {
    .devices = devices,
    .connections = connections,
    .to_connection = to_connection
  };
}

double cluster_t::move(int src, int dst, uint64_t bytes) const {
  connection_t const& c = connections[to_connection.at({src,dst})];
  return (1.0 / c.bandwidth) * bytes;
}

double cluster_t::compute(int loc, uint64_t flops) const {
  device_t const& d = devices[loc];
  return (1.0 / d.compute) * flops;
}

int cluster_t::util(int loc, einsummable_t const& e) const {
  return devices[loc].get_util(e);
}

