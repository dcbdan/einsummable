#pragma once
#include "../base/setup.h"
#include "../einsummable/einsummable.h"

struct cluster_t {
  struct device_t {
    device_t(uint64_t compute);

    device_t(
      int capacity,
      std::function<tuple<int,double>(einsummable_t const&)> f_cost);

    int capacity;     // the number of workers

    // the amount of capacity and time a kernel uses up
    std::function<tuple<int,double>(einsummable_t const&)> f_cost;
  };

  struct connection_t {
    uint64_t bandwidth; // bytes per second
    int src;
    int dst;

    tuple<int,int> key() const { return {src,dst}; }
  };

  static cluster_t make(vector<device_t> const& ds, vector<connection_t> const& cs);

  connection_t const& get_connection(int src, int dst) const {
    return connections[to_connection.at({src,dst})];
  }

  double move(int src, int dst, uint64_t bytes) const;

  tuple<int, double> compute(int loc, einsummable_t const& e) const;

  vector<device_t> const devices;
  vector<connection_t> const connections;
  map<tuple<int, int>, int> const to_connection;

  int num_device() const { return devices.size(); }
};
