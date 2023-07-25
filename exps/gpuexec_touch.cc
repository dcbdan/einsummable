#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"

#include "../src/einsummable/scalarop.h"

#include "../src/execution/gpu/gpu_kernel_manager.h"




touch_t example_touch(dtype_t dtype) {
  touchdim_t dm1 = {10,10,3,5,5};
  //touchdim_t dm2 = {10,10,5,4,5};
  //touchdim_t dm3 = {10,10,3,5,5};
  //touchdim_t dm4 = {10,10,1,2,5};
  vector<touchdim_t> selection;
  selection.push_back(dm1);
  //selection.push_back(dm2);
  //selection.push_back(dm3);
  //selection.push_back(dm4);

  //optional<castable_t>()
  //castable_t::mul
  //castable_t::add
  //castable_t::min
  //castable_t::max

  return touch_t {
    .selection = selection,
    .castable = castable_t::mul,
    .dtype = dtype
  };
}


int main(){

    dtype_t dtype = dtype_t::f16;

    const int totalSize = 10 * 10;

    touch_t exp_touch = example_touch(dtype);


    dbuffer_t out_ref = make_dbuffer(dtype, totalSize);
    dbuffer_t out = make_dbuffer(dtype, totalSize);
    dbuffer_t inn = make_dbuffer(dtype, totalSize);


    out.iota(1);
    out_ref.iota(1);
    inn.random("-2.0", "2.0");

    reference_touch(exp_touch, out_ref, inn);

    size_t sizeOut = out.size();
    size_t sizeIn = inn.size();

    void *in,  *ou;
    cudaMalloc((void**)&in, sizeIn);
    cudaMalloc((void**)&ou, sizeOut);

    cudaMemcpy(ou, out.ptr(), sizeOut, cudaMemcpyHostToDevice);
    cudaMemcpy(in, inn.ptr(), sizeIn, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //auto built_gpu = build_touch(exp_touch);

    //built_gpu(stream,ou,in);

    kernel_manager_t km;

    km(exp_touch,stream,ou,in);


    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    cudaMemcpy(out.ptr(), ou,sizeOut, cudaMemcpyDeviceToHost);


    if(!is_close(out_ref, out,0.01f)) {\
 
      DOUT(out_ref);
      DOUT(out);
      printf("Touch operation is FAIL\n");
    }else{
      std::cout << "Touch Operation is a suecessful!" <<std::endl;
    }









}