#include "storage.h"

void storage_manager_t::open_file()
{
	file.open(file_name, std::ios::binary | std::ios::out | std::ios::in);
	if (!file) throw std::runtime_error("Error in open_file method. Failed to open the file: " + file_name + ".");
}

void storage_manager_t::write_to_disk(const buffer_t& buffer, const memloc_t& src, const stoloc_t& dst)
{
	open_file();
	auto const& block = find_first_available(src.size);
	uint64_t position;
	if (block == blocks.end())
	{
		file.seekp(0, std::ios::end);
		allocate_block(block, src.size);
	} 
	else
	{
		file.seekp(block->beg, std::ios::beg);
	}

	position = file.tellp();

	file.write((char*)buffer->raw() + src.offset, src.size);
	if (file.fail()) throw std::runtime_error("Error in write_to_disk method. Evicting data to disk unsuccessful.");
	tensors[dst.id] = position;
	
	file.close();
}

void storage_manager_t::read_from_disk(const buffer_t& buffer, const stoloc_t& src, const memloc_t& dst)
{
	open_file();
	uint64_t position = tensors[src.id];
	file.seekg(position, std::ios::beg);

	file.read((char*)buffer->raw() + dst.offset, dst.size);
	if (file.fail()) throw std::runtime_error("Error in read_from_disk. Loading data from disk unsuccessful.");
	tensors.erase(src.id);
	create_free_space(position, dst.size);

	file.close();
}

void storage_manager_t::evict(const evict_t& op, const buffer_t& buffer)
{
	write_to_disk(buffer, op.src, op.dst); 
}

void storage_manager_t::load(const load_t& op, const buffer_t& buffer)
{
	read_from_disk(buffer, op.src, op.dst);
}

vector<block_t>::iterator storage_manager_t::find_first_available(uint64_t size)
{
	using return_t = std::tuple<iter_t, iter_t, uint64_t>>;

	for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter)
	{
		iter_t ret  = iter;
		if (size <= iter->size()) 
		{
			break;
		}
	}

	return iter;
}

void storage_manager_t::create_free_space(uint64_t position, uint64_t size)
{
	for (auto iter = blocks.begin(); iter != blocks.end(); ++iter)
	{
		if (iter->end == position)
		{
			iter->end = position+size;
			return;
		}
	}

	block_t block = {position, position + size};
	blocks.push_back(block);
}

void storage_manager_t::allocate_block(vector<block_t>::iterator block, uint64_t size)
{
	if (size == block->size()) {
		blocks.erase(block);
		return;
	}

	block->beg += size;
}