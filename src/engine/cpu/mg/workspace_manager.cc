#include "workspace_manager.h"

int cpu_workspace_manager_t::acquire(uint64_t size)
{
    int ret = -1;
    int smallest;
    for (int i = 0; i != items.size(); ++i) {
        auto const& [is_available, buffer] = items[i];
        uint64_t const& sz = buffer->size;
        if (is_available && size <= sz) {
            if (ret == -1) {
                ret = i;
                smallest = sz;
            } else {
                if (sz < smallest) {
                    ret = i;
                    smallest = sz;
                }
            }
        }
    }
    if (ret == -1) {
        items.emplace_back(false, make_buffer(size));
        ret = items.size() - 1;
    } else {
        auto& [is_available, _] = items[ret];
        is_available = false;
    }

    return ret;
}

optional<cpu_workspace_resource_t>
cpu_workspace_manager_t::try_to_acquire_impl(uint64_t const& size)
{
    std::unique_lock lk(m_items);

    cpu_workspace_resource_t ret;

    ret.which = acquire(size);

    auto& [_, buffer] = items[ret.which];

    ret.ptr = buffer->raw();
    ret.size = buffer->size;

    return ret;
}

void cpu_workspace_manager_t::release(int which)
{
    std::unique_lock lk(m_items);
    auto& [is_available, _] = items[which];
    is_available = true;
}
