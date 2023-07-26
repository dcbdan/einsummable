#include "storage.h"

storage_t::storage_t(string filename) : file_name(filename)
{
	open_file();
}

void storage_t::open_file()
{
	file.open(file_name, std::ios::binary | std::ios::out | std::ios::in);
	if (!file)
	{
		throw std::runtime_error("Failed to open the file: " + file_name + ".");
	}
}

void storage_t::write(const buffer_t& buffer, int id)
{
	auto const& block = find_first_available(buffer->size);
	uint64_t position;
	if (block == blocks.end())
	{
		file.seekp(0, std::ios::end);
		allocate_block(block, buffer->size);
	} 
	else
	{
		file.seekp(block->beg, std::ios::beg);
	}

	position = file.tellp();

	file.write((char*)buffer->raw(), buffer->size);
	if (file.fail())
 	{
		throw std::runtime_error("Evicting data to disk unsuccessful.");
	} 

	tensors[id] = block_t(position, position + buffer->size);
}

void storage_t::read(const buffer_t& buffer, int id)
{
  block_t tensor = tensors[id];
  file.seekg(tensor.beg, std::ios::beg);

  file.read((char*)buffer->raw(), buffer->size);
  if (file.fail()) 
	{
		throw std::runtime_error("Loading data from disk unsuccessful.");
	}

  tensors.erase(id);
  create_free_space(tensor.beg, buffer->size);
}

void storage_t::remove(int id)
{
	block_t tensor = tensors[id];
	tensors.erase(id);
	create_free_space(tensor.beg, tensor.size());
}

vector<storage_t::block_t>::iterator storage_t::find_first_available(uint64_t size)
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
	if(try_merge_blocks(position, size)) { return; }

	block_t block = {position, position + size};
	blocks.emplace_back(block);
}

/*
	Method that checks for adjacent blocks of currently loaded block and 
	merges them in one bigger block if possible 

	So if we have |_______|xxxxxxx|___| this situation where x is memory to be freed, 
	this would become |_________________| -> one bigger block
*/
bool storage_t::try_merge_blocks(uint64_t position, uint64_t size)
{
	for (auto iter = blocks.begin(); iter != blocks.end(); ++iter)
	{
		auto preceding = iter;
		auto following = iter + 1;
		if (can_be_merged(preceding->end, position))
		{
			preceding->end = position+size;
			return true;
		}
		
		if (following == blocks.end()) { return false; }
		
		if (can_be_merged(following->beg, position+size))
		{
			following->beg = position;
			return true;
		}

		if (can_be_merged(preceding->end, position) && can_be_merged(following->end, position + size))
		{
			preceding->end = following->end;
			blocks.erase(following);
			return true;
		}
	}

	return false;
}

bool can_be_merged(uint64_t first_position, uint64_t last_position)
{
	return first_position == last_position;
} 



void storage_t::allocate_block(vector<block_t>::iterator block, uint64_t size)
{
	if (size == block->size()) {
		blocks.erase(block);
		return;
	}

	block->beg += size;
}