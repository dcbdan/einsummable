#include "misc.h"

#include <cctype>

struct read_np_array_t {
  enum token_t { real, comma, open, close, eof };

  read_np_array_t(string const& str)
    : ss(str)
  {
    get();
  }

  token_t const& peek() {
    return _peek;
  }

  token_t get() {
    token_t ret = _peek;
    char c;
    while(!ss.eof() && std::isspace(c = ss.peek())) {
      ss.get();
    }

    if(ss.eof()) {
      _peek = eof;
    } else if(std::isdigit(c) || c == '-') {
      ss >> v;
      reals.push_back(v);
      _peek = real;
    } else if(c == ',') {
      ss.get();
      _peek = comma;
    } else if(c == '[') {
      ss.get();
      _peek = open;
    } else if(c == ']') {
      ss.get();
      _peek = close;
    } else {
      throw std::runtime_error("read_np_array_t tokenizer error");
    }

    return ret;
  }

  void expect(token_t t) {
    expect(vector<token_t>{t});
  }
  token_t expect(vector<token_t> const& ts) {
    for(token_t const& t: ts) {
      if(t == peek()) {
        get();
        return t;
      }
    }
    throw std::runtime_error("parse expect error");
  }

  vector<uint64_t> operator()() {
    expect(open);
    if(peek() == real) {
      uint64_t cnt = 1;
      get();
      // comma then real until close
      while(comma == expect({comma, close})) {
        expect(real);
        cnt++;
      }
      return {cnt};
    } else {
      vector<uint64_t> shape = (*this)();
      uint64_t cnt = 1;
      while(comma == expect({comma, close})) {
        vector<uint64_t> _shape = (*this)();
        if(shape != _shape) {
          throw std::runtime_error("parse error: not a hyperrectangle");
        }
        cnt++;
      }
      shape.insert(shape.begin(), cnt);
      return shape;
    }
  }

  vector<double> reals;
private:
  double v;
  token_t _peek;
  std::stringstream ss;
};

tuple<dbuffer_t, vector<uint64_t>>
read_array(dtype_t dtype, string const& str) {
  if(dtype_is_complex(dtype)) {
    throw std::runtime_error("invalid dtype");
  }

  read_np_array_t reader(str);

  vector<uint64_t> shape = reader();
  vector<double> const& data = reader.reals;

  if(data.size() != product(shape)) {
    throw std::runtime_error("incorrect parse");
  }

  dbuffer_t ret = make_dbuffer(dtype, data.size());

  if(dtype == dtype_t::f16) {
    auto* ptr = ret.f16();
    std::copy(data.begin(), data.end(), ptr);
  } else if(dtype == dtype_t::f32) {
    auto* ptr = ret.f32();
    std::copy(data.begin(), data.end(), ptr);
  } else if(dtype == dtype_t::f64) {
    auto* ptr = ret.f64();
    std::copy(data.begin(), data.end(), ptr);
  }

  return {ret, shape};
}
