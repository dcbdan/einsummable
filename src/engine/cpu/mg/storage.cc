#include "storage.h"

#include <stdlib.h>
#include <unistd.h>

// stolen from
//   https://stackoverflow.com/questions/499636/
//     how-to-create-a-stdofstream-to-a-temp-file
string open_temp(string path, std::fstream& f)
{
    path += "/es_XXXXXX";
    std::vector<char> dst_path(path.begin(), path.end());
    dst_path.push_back('\0');

    int fd = mkstemp(&dst_path[0]);
    if (fd != -1) {
        path.assign(dst_path.begin(), dst_path.end() - 1);
        f.open(path.c_str(), std::ios_base::in | std::ios_base::out | std::ios_base::binary);
        close(fd);
    }
    return path;
}

cpu_storage_t::cpu_storage_t()
{
    filename = open_temp("/tmp", file);
}

cpu_storage_t::stoalloc_t::stoalloc_t()
    : allocator(std::numeric_limits<uint64_t>::max(),
                allocator_settings_t{.strat = allocator_strat_t::first, .alignment_power = 0})
{
}

uint64_t cpu_storage_t::stoalloc_t::allocate(uint64_t sz)
{
    auto maybe = allocator.allocate(sz);
    if (maybe) {
        return std::get<0>(maybe.value());
    }
    throw std::runtime_error(
        "allocator_t could not allocate in cpu_storage_t::stoalloc_t::allocate()");
}

void cpu_storage_t::stoalloc_t::free(uint64_t offset)
{
    allocator.free(offset, 0);
}

uint64_t cpu_storage_t::stoalloc_t::get_size_at(uint64_t offset) const
{
    auto maybe = allocator.get_allocated_region(offset);
    if (!maybe) {
        throw std::runtime_error("get_size_at: this memory not allocated");
    }
    auto const& [beg, end] = maybe.value();
    if (beg != offset) {
        throw std::runtime_error("get_size_at: offset provided in middle of allocated region");
    }
    return end - beg;
}

void cpu_storage_t::write(buffer_t buffer, int id)
{
    std::unique_lock lk(m);

    auto iter = offsets.find(id);
    if (iter != offsets.end()) {
        throw std::runtime_error("id already in storage; try removing first");
    }

    // TODO: what if the id is already here?
    uint64_t offset = allocator.allocate(buffer->size);
    offsets.insert({id, offset});

    file.seekp(offset, std::ios::beg);
    char* data = reinterpret_cast<char*>(buffer->data);
    file.write(data, buffer->size);

    file.flush();
}

void cpu_storage_t::read(buffer_t buffer, int id)
{
    std::unique_lock lk(m);

    auto iter = offsets.find(id);
    if (iter == offsets.end()) {
        throw std::runtime_error("read: id not in storage");
    }
    uint64_t const& offset = iter->second;
    uint64_t        sz = allocator.get_size_at(offset);
    if (sz != buffer->size) {
        throw std::runtime_error("storage read: incorrect size for input buffer");
    }
    _read(buffer, offset);
}

buffer_t cpu_storage_t::read(int id)
{
    std::unique_lock lk(m);

    auto iter = offsets.find(id);
    if (iter == offsets.end()) {
        throw std::runtime_error("read: id not in storage");
    }
    uint64_t const& offset = iter->second;
    uint64_t        sz = allocator.get_size_at(offset);
    buffer_t        buffer = make_buffer(sz);
    _read(buffer, offset);

    return buffer;
}

void cpu_storage_t::_read(buffer_t buffer, uint64_t offset)
{
    file.seekg(offset, std::ios::beg);
    char* data = reinterpret_cast<char*>(buffer->data);
    file.read(data, buffer->size);
    file.flush();
}

void cpu_storage_t::remove(int id)
{
    std::unique_lock lk(m);
    _remove(id);
}

void cpu_storage_t::_remove(int id)
{
    auto iter = offsets.find(id);
    if (iter == offsets.end()) {
        throw std::runtime_error("storage remove: id not in storage");
    }
    uint64_t const& offset = iter->second;
    allocator.free(offset);
    offsets.erase(iter);
}

void cpu_storage_t::remap(vector<std::array<int, 2>> const& old_to_new_stoids)
{
    map<int, int> items;
    for (auto const& old_new : old_to_new_stoids) {
        items.insert({old_new[0], old_new[1]});
    }
    remap(items);
}

void cpu_storage_t::remap(map<int, int> const& mm)
{
    std::unique_lock lk(m);

    map<int, uint64_t> new_offsets;
    set<int>           will_remove;

    for (auto const& [id, offset] : offsets) {
        auto iter = mm.find(id);
        if (iter == mm.end()) {
            will_remove.insert(id);
        } else {
            new_offsets.insert({iter->second, offset});
        }
    }

    for (int const& id : will_remove) {
        _remove(id);
    }

    offsets = new_offsets;
}

int cpu_storage_t::get_max_id() const
{
    if (offsets.size() == 0) {
        return -1;
    }
    return offsets.rbegin()->first;
}
