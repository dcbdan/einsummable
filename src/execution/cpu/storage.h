#pragma once 
#include <unordered_map>
#include <fstream>
#include "../../base/setup.h"
#include "../../base/buffer.h"
#include "../../einsummable/memgraph.h"
#include <shared_mutex>
#include <mutex>

struct storage_t 
{
	void write(const buffer_t& buffer, int id);
	void read (const buffer_t& buffer, int id);
	void remove(int id);

	storage_t(const string filename);
	storage_t(const storage_t& other);
private: 
	
	void open_file();

	struct block_t {
		uint64_t beg;
		uint64_t end;

		uint64_t size() const { return end - beg; }
		block_t() {}
		block_t(uint64_t begin, uint64_t end) : beg(begin), end(end) {}
	};

	std::unordered_map<int, block_t> tensors;
	std::fstream file;
	string file_name;
	std::shared_mutex mtx;

	vector<block_t> blocks;
	using iter_t = vector<block_t>::iterator;
	
	vector<block_t>::iterator find_first_available(uint64_t size);

	bool try_merge_blocks(uint64_t position, uint64_t size);
	void create_free_space(uint64_t position, uint64_t size);
	void allocate_block(vector<block_t>::iterator block, uint64_t size);
	bool can_be_merged(int64_t first_position, uint64_t last_position);
};
