#include <string>
#include <iostream>

#include "sentencepiece_processor.h"

int main(int argc, char** argv) {
  sentencepiece::SentencePieceProcessor processor;
  const auto status = processor.Load(string(argv[0]));
  if (!status.ok()) {
     std::cerr << status.ToString() << std::endl;
     return 0;
     // error
  }

  std::vector<std::string> pieces;
  processor.Encode("This is a test.", &pieces);
  for (const std::string &token : pieces) {
    std::cout << token << std::endl;
  }
}
