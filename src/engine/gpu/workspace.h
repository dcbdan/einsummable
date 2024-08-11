#pragma once
#include "../../base/setup.h"

#include "utility.h"

#include "../resource_manager.h"

#include <cstdint>
#include <cuda_runtime.h>

#include <mutex>

struct gpu_workspace_desc_t {
    int      device;
    uint64_t size;
};

struct gpu_workspace_resource_t {
    int      device;
    void*    ptr;
    uint64_t size;

    tuple<void*, uint64_t> as_tuple() const
    {
        return {ptr, size};
    }
};

struct gpu_workspace_manager_t : rm_template_t<gpu_workspace_desc_t, gpu_workspace_resource_t> {
    gpu_workspace_manager_t();

    ~gpu_workspace_manager_t();

private:
    std::mutex m;

    vector<vector<tuple<void*, uint64_t>>> data;

    optional<gpu_workspace_resource_t> try_to_acquire_impl(gpu_workspace_desc_t const& desc);

    void release_impl(gpu_workspace_resource_t const& resource);
};
