#pragma once
#include "setup.h"

namespace timeplot_ns {

using std::endl;

int double_to_int(double v);

template <typename T>
struct quote_t {
  quote_t(T const& v): val(v) {}

  T const& val;
};

template <typename T>
quote_t<T> quote(T const& t) {
  return quote_t<T>(t);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, quote_t<T> const& q) {
  out << "\"" << q.val << "\"";
  return out;
}

struct box_t {
  int row;
  double start;
  double stop;
  string text;
};

std::ostream& operator<<(std::ostream& out, box_t const& box);

tuple<int,int,int,int> xywh(
  tuple<int,int> const& top_left,
  tuple<int,int> const& bot_right);

struct svg_box_t {
  tuple<int,int> top_left;
  tuple<int,int> bot_right;

  tuple<int,int> rxy;

  string fill_color;
  string stroke_color;
  int stroke_width;
  double opacity;
};

std::ostream& operator<<(std::ostream& out, svg_box_t const& box);

svg_box_t row_background(
  int which_row,
  int width,
  int row_height);

struct svg_text_t {
  tuple<int,int> top_left;
  tuple<int,int> bot_right;
  string text;
  string fill_color;
};

std::ostream& operator<<(std::ostream& out, svg_text_t const& text);

tuple<svg_box_t, svg_text_t> task_box(
  int which_row, int row_height,
  int start, int finish,
  string const& text);

}

void timeplot(
  std::ostream& out,
  vector<timeplot_ns::box_t> const& boxes,
  int row_height,
  int min_box_width,
  optional<double> actual_makespan = std::nullopt);

