#include "../src/base/setup.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/dbuffer.h"
#include "./modules.h"

#include <iostream>
#include <fstream>

int main() {

  // Open the binary file in read mode
  std::string path_to_binary = "/Users/kiwibird/StudyFile/chrisResearch/einsummable_test/output.bin";
  std::ifstream file(path_to_binary, std::ios::binary);
  
  if (!file) {
    std::cerr << "Failed to open the file." << std::endl;
    return 1;
  }
  // Vector to store the extracted data
  map<string, tuple<uint64_t, vector<float16_t>>> extracted_data;
  
  while (true) {
    // Read the text data (name of weight tensor)
    char text_data[51];
    file.read(text_data, 50);
    text_data[50] = '\0';
    std::string name(text_data);
    
    // If no more data to read, break the loop
    if (name.empty() || file.eof()) {
        break;
    }

    std::string space = " ";
    const auto strEnd = name.find_last_not_of(space);
    const auto strRange = strEnd + 1;
    name = name.substr(0, strRange);
    
    // Read the binary data (size of tensor)
    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    
    // Read the tensor data
    std::vector<float16_t> tensor(size);
    file.read(reinterpret_cast<char*>(tensor.data()), size * sizeof(float16_t));

    // Store the extracted data
    extracted_data.insert({name, {size, tensor}});
  }


  set_default_dtype(dtype_t::f16);
  model_args_t model_args = model_args_t::make_default();
  graph_writer_t writer;
  std::string transformer_name= "name";
  transformer_t model = transformer_t(&writer, transformer_name, model_args);
  auto weight_to_name = model.input_map();
  for (auto const& [weight_id, name]: weight_to_name) {
    std::cout << name << std::endl;
  }
  map<int, dbuffer_t> weight_to_buffer;
  // for (auto const& [weight_id, name]: weight_to_name) {
  //   std::cout << name << std::endl;
  //   dbuffer_t buffer = make_dbuffer(dtype_t::f16, std::get<0>(extracted_data.at(name)));
  //   weight_to_buffer.insert({weight_id, buffer});
  //   vector<float16_t> halfs = std::get<1>(extracted_data.at(name));
  //   std::copy(halfs.begin(), halfs.end(), buffer.f16());
  // }

  return(0);

}