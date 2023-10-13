#pragma once
#include "../base/setup.h"

struct desc_base_t {
  virtual ~desc_base_t() = default;
};
using desc_ptr_t = std::shared_ptr<desc_base_t>;

struct resource_base_t {
  virtual ~resource_base_t() = default;
};
using resource_ptr_t = std::shared_ptr<resource_base_t>;

struct resource_manager_base_t {
  // resource_ptr_t may be a nullptr if did not acquire
  virtual resource_ptr_t try_to_acquire(desc_ptr_t desc) = 0;

  virtual void release(resource_ptr_t resource) = 0;

  // TODO: Should handles exist as part of the base or just as part of
  //       rm_template_t? If resource_manager_t is to hold something that is
  //       not a rm_template_t, then these need to be here.
  virtual bool handles(desc_ptr_t      desc)     const = 0;
  virtual bool handles(resource_ptr_t resource)  const = 0;
};
using rm_ptr_t = std::shared_ptr<resource_manager_base_t>;

template <typename D, typename R>
struct rm_template_t : resource_manager_base_t {
  struct desc_t : desc_base_t {
    desc_t(D const& x): d(x) {}
    D d;
  };
  struct resource_t : resource_base_t {
    resource_t(R const& x): r(x) {}
    R r;
  };
  // Note: Both desc_t and resource_t will copy D and R in their constructor.
  //       The assumption is that D and R are lightweight

  bool handles(desc_ptr_t desc) const {
    desc_t* maybe = dynamic_cast<desc_t*>(desc.get());
    return bool(maybe);
  }
  bool handles(resource_ptr_t resource) const {
    resource_t* maybe = dynamic_cast<resource_t*>(resource.get());
    return bool(maybe);
  }

  resource_ptr_t try_to_acquire(desc_ptr_t desc) {
    D const& d = static_cast<desc_t*>(desc.get())->d;
    optional<R> maybe = try_to_acquire_impl(d);
    if(maybe) {
      return resource_ptr_t(new resource_t(maybe.value()));
    } else {
      return resource_ptr_t(nullptr);
    }
  }
  void release(resource_ptr_t resource) {
    R const& r = get_resource(resource);
    release_impl(r);
  }
  // Note: Both try_to_acquire and resource_ptr_t are making the assumption
  //       that the desc_ptr_t and resource_ptr_t provided will return true
  //       when given to handles. (Hence the static cast)

  static
  desc_ptr_t make_desc(D const& d) {
    return desc_ptr_t(new desc_t(d));
  }

  static
  R const& get_resource(resource_ptr_t resource) {
    return static_cast<resource_t*>(resource.get())->r;
  }

private:
  virtual optional<R> try_to_acquire_impl(D const& d) = 0;
  virtual void release_impl(R const& r) = 0;
};

struct resource_manager_t
  : rm_template_t<vector<desc_ptr_t>, vector<resource_ptr_t>>
{
  resource_manager_t(vector<rm_ptr_t> const& ms)
    : managers(ms)
  {}

private:
  optional<vector<resource_ptr_t>> try_to_acquire_impl(vector<desc_ptr_t> const& descs);
  void release_impl(vector<resource_ptr_t> const& resources);

  resource_ptr_t try_to_acquire_unit(desc_ptr_t desc);
  void release_unit(resource_ptr_t resource);

  vector<rm_ptr_t> managers;
};

