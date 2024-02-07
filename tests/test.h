#pragma once
#include "../src/base/setup.h"

#include <memory>

struct tests_t {
  tests_t();

  tests_t(string name);

  using run_test_t = std::function<optional<string>()>;

private:
  struct test_t {
    string name;
    std::function<optional<string>()> run;
  };

  struct node_t;

  using node_ptr_t = std::unique_ptr<node_t>;

  struct node_t {
    node_t(string name);

    void insert_test(string name, run_test_t run);

    node_t& insert_node(string name);

    using child_t = std::variant<node_ptr_t, test_t>;

    void run(string base) const;

  private:
    static string const& get_child_name(child_t const& child);

    void check_name(string name) const;

    string name;
    vector<child_t> children;
  };

public:
  struct context_t {
    context_t make_context(string name);

    void insert_test(string name, run_test_t run);

  private:
    friend class tests_t;

    context_t(node_t& node);

    node_t& node;
  };

  context_t make_context(string name);

  void run();

  static bool is_valid_name(string name);

  std::unique_ptr<node_t> node;
};
