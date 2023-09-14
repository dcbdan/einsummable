#pragma once
#include "../../base/setup.h"

struct numa_info_t {
  numa_info_t();
  
  static void pin_to_thread(int i);

  static void print();

  void pin_to_this_numa_thread(int n) const;

  void pin_to_this_numa() const;

  static void unpin();

private:
  vector<unsigned int> which;
  unsigned int num_numa;
};

numa_info_t const& get_numa_info();
