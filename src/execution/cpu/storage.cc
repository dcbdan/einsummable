#include "storage.h"

storage_t::storage_t(string filename) : file_name(filename)
{
	open_file();
}

storage_t::~storage_t() 
{
	if (file.is_open())
	{
		file.close();
	}
}

void storage_t::open_file()
{
	file.open(file_name, std::ios::binary | std::ios::out | std::ios::in);
	if (!file)
	{
		throw std::runtime_error("Failed to open the file: " + file_name + ".");
	}
}

void storage_t::write_to_disk(const buffer_t& buffer, const memloc_t& src, const stoloc_t& dst)
{
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
	if (file.fail())
 	{
		throw std::runtime_error("Evicting data to disk unsuccessful.");
	} 
	tensors[dst.id] = position;
}

void storage_t::read_from_disk(const buffer_t& buffer, const stoloc_t& src, const memloc_t& dst)
{
  uint64_t position = tensors[src.id];
  file.seekg(position, std::ios::beg);

  file.read((char*)buffer->raw() + dst.offset, dst.size);
  if (file.fail()) 
	{
		throw std::runtime_error("Loading data from disk unsuccessful.");
	}
  tensors.erase(src.id);
  create_free_space(position, dst.size);
}

vector<block_t>::iterator storage_t::find_first_available(uint64_t size)
{
  iter_t ret = blocks.end(); // just to make sure it has initial value

  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter)
  {
    ret = iter;
    if (size <= iter->size()) 
	  {
	    break;
	  }
	}

	return ret;
}

void storage_t::create_free_space(uint64_t position, uint64_t size)
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

void storage_t::allocate_block(vector<block_t>::iterator block, uint64_t size)
{
	if (size == block->size()) {
		blocks.erase(block);
		return;
	}

	block->beg += size;
}

void storage_t::evict(const evict_t& op, const buffer_t& buffer)
{
  write_to_disk(buffer, op.src, op.dst); 
}

void storage_t::load(const load_t& op, const buffer_t& buffer)
{
  read_from_disk(buffer, op.src, op.dst);
}

void storage_t::save(const memloc_t src, const stoloc_t dst, buffer_t& buffer)
{
	write_to_disk(buffer, src, dst);
}

void storage_t::load(const stoloc_t src, const memloc_t dst, buffer_t& buffer)
{
	read_from_disk(buffer, src, dst);
}