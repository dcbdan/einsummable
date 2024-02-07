#include "test.h"

tests_t::tests_t()
  : tests_t("tests")
{}

tests_t::tests_t(string name)
  : node(std::make_unique<node_t>(name))
{}

tests_t::node_t::node_t(string name)
  : name(name)
{}

void tests_t::node_t::insert_test(string name, tests_t::run_test_t run)
{
  check_name(name);
  children.push_back(test_t { .name = name, .run = run });
}

tests_t::node_t& tests_t::node_t::insert_node(string name)
{
  check_name(name);
  children.push_back(std::make_unique<node_t>(name));
  return *std::get<node_ptr_t>(children.back());
}

void tests_t::node_t::run(string base) const
{
  if(base.size() > 0) {
    DOUT(base + "/");
  }
  for(auto const& child: children) {

    string name =
      base.size() == 0       ?
                   get_child_name(child) :
      base + "/" + get_child_name(child) ;

    if(std::holds_alternative<node_ptr_t>(child)) {
      std::get<node_ptr_t>(child)->run(name);
    } else if(std::holds_alternative<test_t>(child)) {
      auto const& f = std::get<test_t>(child).run;
      auto maybe_error = f();
      if(maybe_error) {
        throw std::runtime_error(
          "failed test: " + name + "\n" + "reason: " +
          maybe_error.value());
      }
      DOUT(name);
    } else {
      throw std::runtime_error("should not reach");
    }
  }
}

string const& tests_t::node_t::get_child_name(child_t const& child)
{
  if(std::holds_alternative<node_ptr_t>(child)) {
    return std::get<node_ptr_t>(child)->name;
  } else if(std::holds_alternative<test_t>(child)) {
    return std::get<test_t>(child).name;
  } else {
    throw std::runtime_error("should not reach");
  }
}

void tests_t::node_t::check_name(string name) const {
  if(!tests_t::is_valid_name(name)) {
    throw std::runtime_error("check_name: tests does not allow this name");
  }
  for(child_t const& child: children) {
    if(get_child_name(child) == name) {
      throw std::runtime_error("check_name: already have this name");
    }
  }
}

tests_t::context_t
tests_t::context_t::make_context(string name) {
  node_t& ret = node.insert_node(name);
  return context_t(ret);
}

void tests_t::context_t::insert_test(string name, run_test_t run) {
  node.insert_test(name, run);
}

tests_t::context_t::context_t(node_t& node)
  : node(node)
{}

tests_t::context_t
tests_t::make_context(string name) {
  node_t& ret = node->insert_node(name);
  return context_t(ret);
}

void tests_t::run() {
  node->run("");
}

bool tests_t::is_valid_name(string name) {
  // TODO: implement this
  return true;
}
