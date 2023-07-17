#include "storage.h"

void storage_manager_t::open_file()
{
    file.open(file_name, std::ios::binary | std::ios::out | std::ios::in);
    if (!file) throw std::runtime_error("Error in open_file method. Failed to open the file: " + file_name + ".");
}

void storage_manager_t::write_to_disk(const buffer_t& buffer, const memloc_t& src, const stoloc_t& dst)
{
    open_file();
    file.seekp(0, std::ios::end);
    uint64_t position = file.tellp();

    file.write((char*)buffer->raw() + src.offset, src.size);
    if (file.fail()) throw std::runtime_error("Error in write_to_disk method. Evicting data to disk unsuccessful.");
    tensors[dst.id] = position;

    file.close();
}

void storage_manager_t::read_from_disk(const buffer_t& buffer, const stoloc_t& src, const memloc_t& dst)
{
    open_file();
    uint64_t position = tensors[src.id];
    file.seekg(positiong, std::ios::beg);

    file.read((char*)buffer->raw() + dst.offset, dst.size);
    if (file.fail()) throw std::runtime_error("Error in read_from_disk. Loading data from disk unsuccessful.");
    tensors.erase(src.id);

    file.close();
}

void storage_manager_t::evict(const evict_t& op, const buffer_t& buffer)
{
    try 
    {
        write_to_disk(buffer, op.src, op.dst);
    }
    catch (std::runtime_error e)
    {
        DOUT(e.what());
    }   
}

void storage_manager_t::load(const load_t& op, const buffer_t& buffer)
{
    try
    {
        read_from_disk(buffer, op.src, op.dst);
    }
    catch(std::runtime_error e)
    {
        DOUT(e.what());
    }
}