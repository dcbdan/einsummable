#include "setup.h"

#include <type_traits>

map<string, string> make_argstrs(int argc, char** argv) {
  if(argc % 2 != 0) {
    throw std::runtime_error("argstrs needs even number of args");
  }
  map<string, string> ret;
  for(int i = 0; i != argc; i += 2) {
    ret.insert({argv[i], argv[i+1]});
  }
  return ret;
}

struct args_t {
  args_t(int argc, char** argv)
    : argstrs(make_argstrs(argc-1, argv+1))
  {}

  void insert_arg(string key, string value) {
    argstrs.insert_or_assign(key, value);
  }

  template <typename T>
  T get(string key) const {
    auto iter_argstrs = argstrs.find(key);
    if(iter_argstrs == argstrs.end()) {
      return get_default<T>(key);
    } else {
      string const& val = iter_argstrs->second;
      if constexpr(std::is_same<T, bool>::value) {
        if(val == "true") {
          return true;
        } else if(val == "false") {
          return false;
        } else if(val == "0") {
          return false;
        } else if(val == "1") {
          return true;
        } else {
          throw std::runtime_error("could not parse bool value");
        }
      } else {
        return parse_with_ss<T>(iter_argstrs->second);
      }
    }
  }

  template <typename T>
  T get_default(string key) const
  {
    if constexpr (std::is_same<T, int>::value) {
      return d_int.at(key);
    } else if constexpr (std::is_same<T, int64_t>::value) {
      return d_int64_t.at(key);
    } else if constexpr (std::is_same<T, uint64_t>::value) {
      return d_uint64_t.at(key);
    } else if constexpr (std::is_same<T, float>::value) {
      return d_float.at(key);
    } else if constexpr (std::is_same<T, double>::value) {
      return d_double.at(key);
    } else if constexpr (std::is_same<T, string>::value) {
      return d_string.at(key);
    } else if constexpr (std::is_same<T, bool>::value) {
      return d_bool.at(key);
    } else {
      throw std::runtime_error("get_default: unsupported default type");
    }
  }

  template <typename T>
  void set_default(string key, T const& value) {
    if constexpr (std::is_same<T, int>::value) {
      d_int.insert_or_assign(key, value);
    } else if constexpr (std::is_same<T, int64_t>::value) {
      d_int64_t.insert_or_assign(key, value);
    } else if constexpr (std::is_same<T, uint64_t>::value) {
      d_uint64_t.insert_or_assign(key, value);
    } else if constexpr (std::is_same<T, float>::value) {
      d_float.insert_or_assign(key, value);
    } else if constexpr (std::is_same<T, double>::value) {
      d_double.insert_or_assign(key, value);
    } else if constexpr (std::is_same<T, string>::value) {
      d_string.insert_or_assign(key, value);
    } else if constexpr (std::is_same<T, bool>::value) {
      d_bool.insert_or_assign(key, value);
    } else {
      throw std::runtime_error("set_default: unsupported default type");
    }
  }

  void set_default(string key, const char* value) {
    d_string.insert_or_assign(key, value);
  }

private:
  map<string, string   > argstrs;

  map<string, int      > d_int;
  map<string, int64_t  > d_int64_t;
  map<string, uint64_t > d_uint64_t;
  map<string, float    > d_float;
  map<string, double   > d_double;
  map<string, string   > d_string;
  map<string, bool     > d_bool;
};


