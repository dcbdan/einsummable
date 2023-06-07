#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"

touch_t example_touch() {
  touchdim_t dm1 = {10,10,3,5,5};
  touchdim_t dm2 = {10,10,5,4,5};
  touchdim_t dm3 = {10000,10000,598,1972,5000};
  touchdim_t dm4 = {10,10,1,2,5};
  vector<touchdim_t> selection;
  selection.push_back(dm1);
  selection.push_back(dm2);
  selection.push_back(dm3);
  selection.push_back(dm4);

  return touch_t {
    .selection = selection,
    .castable = optional<castable_t>()
  };
}



int main(){
    const int totalSize = 10 * 10 * 10000 * 10;
    float* array = new float[totalSize];

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < totalSize; ++i) {
        array[i] = static_cast<float>(std::rand()) / RAND_MAX * 100.0f;
    }

    float* out1 = new float[totalSize]();
    float* out2 = new float[totalSize]();

    float* d_in;
    float* d_out;

    cudaMalloc(&d_in, totalSize  * sizeof(float));
    cudaMalloc(&d_out, totalSize * sizeof(float));

    cudaMemcpy(d_in, array, totalSize  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out1, totalSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    touch_t exp_touch = example_touch();
    auto built_gpu = build_touch(exp_touch);

    built_gpu(stream,d_out,d_in);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream); 

    cudaDeviceSynchronize();

    cudaMemcpy(out1, d_out, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);



}