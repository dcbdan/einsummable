#include "sentencepiece_processor.h"

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

void print_vector(vector<int> const& xs)
{
    if (xs.size() == 0) {
        cout << "[]";
    } else if (xs.size() == 1) {
        cout << "[" << xs[0] << "]";
    } else {
        cout << "[" << xs[0];
        for (int i = 1; i != xs.size(); ++i) {
            int const& x = xs[i];
            cout << "," << x;
        }
        cout << "]";
    }
}

int main(int argc, char** argv)
{
    // open a debug file
    // std::ofstream debug("debug.txt");
    // debug << "debug" << endl;
    sentencepiece::SentencePieceProcessor processor;
    const auto                            status = processor.Load(argv[1]);
    if (!status.ok()) {
        throw std::runtime_error("could not open file");
    }
    int pad_id = processor.pad_id();
    int vocab_size = processor.GetPieceSize();

    print_vector(vector<int>{pad_id, vocab_size});
    // write vector to debug file
    // debug << pad_id << endl;
    // debug << vocab_size << endl;

    uint64_t    sz;
    vector<int> ret;
    while (true) {
        cin.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        // debug << "/////////////////////" << endl;
        // debug << "reading size:" << sz << endl;
        string str(sz, ' ');
        cin.read(str.data(), sz);
        // debug << "reading string:" << str << endl;

        ret.resize(0);
        auto const status = processor.Encode(str, &ret);
        if (!status.ok()) {
            throw std::runtime_error("could not encode");
        }
        if (ret.size() == 0) {
            throw std::runtime_error("empty number of tokens!");
        }
        print_vector(ret);
        // write vector size to debug file
        // debug << ret.size() << endl;
    }
}

// int main(int argc, char** argv) {
//  sentencepiece::SentencePieceProcessor processor;
//  const auto status = processor.Load(argv[1]);
//  if(!status.ok()) {
//    throw std::runtime_error("could not open file");
//  }
//  int pad_id = processor.pad_id();
//  int vocab_size = processor.GetPieceSize();

// 	cout << pad_id << endl;
// 	cout << vocab_size << endl;

// 	std::ifstream file(argv[2], std::ios_base::in | std::ios_base::binary);
//  using pos_type = std::ifstream::pos_type;
//  vector<pos_type> offsets;
// 	int32_t sz;
//  while(file) {
//    pos_type offset = file.tellg();
//    file.read(reinterpret_cast<char*>(&sz), sizeof(int32_t));
//    if(!file) {
//      break;
//    }
//    if(sz <= 0) {
//      throw std::runtime_error("sz < 0");
//    }
//    offsets.push_back(offset);
//    file.seekg(sz, std::ios_base::cur);
//  }

// 	file.clear();

// 	cout << offsets.size() << endl;

// 	int which;
// 	vector<int> ret;
// 	while(true) {
// 		cin >> which;
//    if(which < 0 || which >= offsets.size()) {
//      throw std::runtime_error("invalid which");
//    }
// 		std::cout << "which " << which << std::endl;
//    auto const& offset = offsets[which];
// 		std::cout << "offset " << offset << std::endl;
//    file.seekg(offset, std::ios_base::beg);
//    file.read(reinterpret_cast<char*>(&sz), sizeof(int32_t));
//    string str(sz, ' ');
//    file.read(str.data(), sz);
// 		std::cout << str << std::endl;
// 		ret.resize(0);
//    auto const status = processor.Encode(str, &ret);
//    if(!status.ok()) {
//      throw std::runtime_error("could not encode");
//    }
//    if(ret.size() == 0) {
//      throw std::runtime_error("empty number of tokens!");
//    }
// 		print_vector(ret);
// 	}
// }
