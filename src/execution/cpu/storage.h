#pragma once 
#include <unordered_map>
#include <fstream>
#include "../../base/setup.h"
#include "../../base/buffer.h"
#include "../../einsummable/memgraph.h"

struct storage_manager_t 
{
    std::unordered_map<int, uint64_t> tensors;
    std::fstream file;
    std::string file_name;

    // Method that encapsulates logic of writing to file inside of storage manager
    void evict(const evict_t& op, const buffer_t& buffer);
    // Method that encapsulates logic of reading from file inside of storage manager
    void load(const load_t&op, const buffer_t& buffer);

    storage_manager_t(const string filename): file_name(filename){}
private: 
    void write_to_disk(const buffer_t& buffer, const memloc_t& src, const stoloc_t& dst);
    void read_from_disk(const buffer_t& buffer, const stoloc_t& src, const memloc_t& dst);
    // Method for safe opening of a file 
    void open_file();
};
