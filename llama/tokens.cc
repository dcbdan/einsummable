#include "../src/base/setup.h"
#include "../src/base/args.h"

#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <sentencepiece_processor.h>

string read_file(string filename) {
  std::ifstream t(filename);
  if(!t.good()) {
    throw std::runtime_error("not opened correctly: " + filename);
  }
  t.seekg(0, std::ios::end);
  auto size = t.tellg();
  if(size == decltype(size)(-1)) {
    throw std::runtime_error("could not get filesize");
  }
  std::string buffer(size, ' ');
  t.seekg(0);
  t.read(&buffer[0], size);
  return buffer;
}

//string read_file(string filename) {
//  std::ifstream t(filename, std::ios_base::in | std::ios_base::binary);
//  if(!t.good()) {
//    throw std::runtime_error("not opened correctly: " + filename);
//  }
//  std::stringstream buffer;
//  buffer << t.rdbuf();
//  return buffer.str();
//}

int main(int argc, char** argv) {
  args_t args(argc, argv);

  string tokenizer = args.get<string>("tokenizer");
  string source    = args.get<string>("source");

  sentencepiece::SentencePieceProcessor processor;
  const auto status = processor.Load(tokenizer);
  if(!status.ok()) {
    throw std::runtime_error("could not open " + write_with_ss(tokenizer));
  }

  int max_filename_len = 0;
  uint64_t total_num_tokens = 0;
  vector<tuple<string, vector<int>>> data;
  for(fs::directory_entry const& filename_: fs::recursive_directory_iterator(source)) {
    if(filename_.is_regular_file()) {
      string filename = filename_.path().string();
      max_filename_len = std::max(max_filename_len, int(filename.size()));
      data.emplace_back(filename, vector<int>{});
      auto& tokens = std::get<1>(data.back());
      auto const status = processor.Encode(read_file(filename), &tokens);
      if(!status.ok()) {
        throw std::runtime_error("could not encode from " + filename);
      }
      total_num_tokens += tokens.size();
    }
  }
  for(auto const& [filename, tokens]: data) {
    DOUT(filename << ": " << string(max_filename_len - filename.size() + 2, ' ') << tokens.size());
  }
  DOUT("total number of tokens: " << total_num_tokens);
}

