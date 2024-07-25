#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/dbuffer.h"

tuple<dbuffer_t, vector<uint64_t>> read_array(dtype_t dtype, string const& str);
