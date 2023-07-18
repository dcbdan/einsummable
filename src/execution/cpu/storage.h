#pragma once 
#include <unordered_map>
#include <fstream>
#include "../../base/setup.h"
#include "../../base/buffer.h"
#include "../../einsummable/memgraph.h"

struct storage_t 
{
	void write(const buffer_t& buffer, int id);
	void read (const buffer_t& buffer, int id);
	void remove(int id);

	storage_t(const string filename): file_name(filename){}
	storage_t(const storage_t& other): file_name(other.file_name){}
private: 
	
	void open_file();

	struct block_t {
		uint64_t beg;
		uint64_t end;

		uint64_t size() const { return end - beg; }
		block_t(uint64_t begin, uint64_t end) : beg(begin), end(end) {}
	};

	std::unordered_map<int, block_t> tensors;
	std::fstream file;
	string file_name;

	vector<block_t> blocks;
	using iter_t = vector<block_t>::iterator;
	
	vector<block_t>::iterator find_first_available(uint64_t size);

	void create_free_space(uint64_t position, uint64_t size);
	void allocate_block(vector<block_t>::iterator block, uint64_t size);
};
