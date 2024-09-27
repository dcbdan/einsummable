#include "../exec_graph.h"
#include "../resource_manager.h"
#include "gpu_kernel_manager.h"
#include <cstdint>

struct gpu_einsummable_t : exec_graph_t::op_base_t {
    gpu_einsummable_t(kernel_manager_t& a, einsummable_t const& b, vector<mem_t> const& c, int d)
        : gpu_km(a), einsummable(b), mems(c), device(d)
    {
    }

    kernel_manager_t& gpu_km;
    einsummable_t     einsummable;
    vector<mem_t>     mems;
    int               device;
    uint64_t          workspace_size;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 2 or 3 resources are always needed for an einsummable
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_einsummable";
    }
};

struct gpu_touch_t : exec_graph_t::op_base_t {
    gpu_touch_t(kernel_manager_t& a, touch_t const& b, int c, vector<mem_t> d, int e)
        : gpu_km(a), touch(b), group_id(c), mems(d), device(e)
    {
    }

    kernel_manager_t& gpu_km;
    touch_t           touch;
    int               group_id;
    vector<mem_t>     mems;
    int               device;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 2 or 3 resources are always needed for a touch
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_touch";
    }
};

struct gpu_copy_t : exec_graph_t::op_base_t {
    gpu_copy_t(memgraph_t::move_t const& m) : move(m) {}

    memgraph_t::move_t move;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 2 resources are always needed for a copy
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_copy";
    }
};

struct gpu_safe_copy_t : exec_graph_t::op_base_t {
    gpu_safe_copy_t(memgraph_t::safe_copy_t const& m) : safe_copy(m) {}

    memgraph_t::safe_copy_t safe_copy;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 2 resources are always needed for a copy
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_safe_copy";
    }
};

// NOTE: Evict and load only support GPU RAM to CPU RAM
// TODO: if additional levels of storage is needed (such as disk)
// then we will need to add new support for that here
struct gpu_evict_t : exec_graph_t::op_base_t {
    gpu_evict_t(memgraph_t::evict_t const& e)
    {
        auto gpu_src = e.src;
        auto cpu_dst = e.dst;
        gpu_offset = gpu_src.offset;
        size = gpu_src.size;
        device = gpu_src.loc;
        storage_id = cpu_dst.as_memsto().get_sto();
    }

    uint64_t gpu_offset;
    uint64_t size;
    int      device;
    int      storage_id;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 1 resource is always needed for an evict
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_evict";
    }
};

struct gpu_load_t : exec_graph_t::op_base_t {
    gpu_load_t(memgraph_t::load_t const& l)
    {
        auto cpu_src = l.src;
        auto gpu_dst = l.dst;
        gpu_offset = gpu_dst.offset;
        size = gpu_dst.size;
        device = gpu_dst.loc;
        storage_id = cpu_src.as_memsto().get_sto();
    }

    uint64_t gpu_offset;
    uint64_t size;
    int      device;
    int      storage_id;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 1 resource is always needed for a load
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_load";
    }
};

struct gpu_constant_t : exec_graph_t::op_base_t {
    gpu_constant_t(kernel_manager_t& a, memgraph_t::constant_t const& c)
        : gpu_km(a), gpu_offset(c.offset), device(c.loc), fill(c.fill.get_constant())
    {
    }

    kernel_manager_t&        gpu_km;
    uint64_t                 gpu_offset;
    int                      device;
    fill_t::constant_t const fill;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 1 resource is always needed for a load
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_constant_fill";
    }
};

struct gpu_lowerTri_t : exec_graph_t::op_base_t {
    gpu_lowerTri_t(kernel_manager_t& a, memgraph_t::constant_t const& l)
        : gpu_km(a), gpu_offset(l.offset), device(l.loc), fill(l.fill.get_lowertri())
    {
    }

    kernel_manager_t&        gpu_km;
    uint64_t                 gpu_offset;
    int                      device;
    fill_t::lowertri_t const fill;

    void launch(resource_ptr_t resource, std::function<void()> callback) const;
    // 1 resource is always needed for a load
    desc_ptr_t resource_description() const;
    void       print(std::ostream& out) const
    {
        out << "gpu_lower_tri_fill";
    }
};
