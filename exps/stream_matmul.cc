#include "stream_matmul.h"
#include <cuda_profiler_api.h>

contraction_stream_t::permute_info_t
contraction_stream_t::permute_info_t::drop_leading(int n) const {
  if(num_leading_modes() < n) {
    throw std::runtime_error("can't drop!");
  }

  vector<int> inn_modes(inn_shape.size() - n);
  std::iota(inn_modes.begin(), inn_modes.end(), n);

  vector<int> out_modes(out_perm.begin() + n, out_perm.end());

  return from_inn_shape(
    vector<uint64_t>(inn_shape.begin() + n, inn_shape.end()),
    inn_modes,
    out_modes);
}

optional<tuple<
  vector<int>, vector<int>, vector<int>, vector<int> >>
contraction_stream_t::make_bs_is_js_ks(
  vector<int> const& lhs_inn_modes,
  vector<int> const& rhs_inn_modes,
  int out_rank)
{
  vector<int> bs;
  vector<int> is;
  vector<int> js;
  vector<int> ks;

  int join_rank = 1 + std::max(
    *std::max_element(lhs_inn_modes.begin(), lhs_inn_modes.end()),
    *std::max_element(rhs_inn_modes.begin(), rhs_inn_modes.end()));

  for(int i = 0; i != join_rank; ++i) {
    bool in_lhs = std::find(
      lhs_inn_modes.begin(), lhs_inn_modes.end(), i) != lhs_inn_modes.end();
    bool in_rhs = std::find(
      rhs_inn_modes.begin(), rhs_inn_modes.end(), i) != rhs_inn_modes.end();
    bool in_out = i < out_rank;

    if(in_lhs && in_rhs && in_out) {
      bs.push_back(i);
    } else if(in_lhs && in_rhs && !in_out) {
      js.push_back(i);
    } else if(in_lhs && !in_rhs && in_out) {
      is.push_back(i);
    } else if(!in_lhs && in_rhs && in_out) {
      ks.push_back(i);
    } else {
      return std::nullopt;
    }
  }

  using ret_t = tuple<vector<int>, vector<int>, vector<int>, vector<int> >;
  return ret_t{bs,is,js,ks};
}

void batch_matmul_gpu(dtype_t dtype,
  uint64_t ni,
  uint64_t nj,
  uint64_t nk,
  bool trans_l,
  bool trans_r,
  void* out,
  void* lhs,
  void* rhs,
  cudaStream_t stream,
  cublasHandle_t cublas_handle,
  uint64_t batch_size) {
    // for now, limiting the dtype to f32. TODO: extending this 
    if (dtype != dtype_t::f32) {
      throw std::runtime_error("dtype must be f32");
    }
    if (!lhs || !rhs || !out) {
      throw std::runtime_error("Input or output pointers cannot be null");
    }
    if (ni <= 0 || nj <= 0 || nk <= 0 || batch_size <= 0) {
      throw std::runtime_error("Invalid matrix dimensions or batch size");
    }

    bool debug = false;

    if (debug) {
      DOUT("lhs matrix: ");
      printFloatGPU(lhs, ni*nj*batch_size);
      DOUT("rhs matrix: ");
      printFloatGPU(rhs, nj*nk*batch_size);
      // DOUT("out matrix: ");
      // printFloatGPU(out, ni*nk*batch_size);
    }

  // Transpose flags to cuBLAS op types
    cublasOperation_t op_l = trans_l ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_r = trans_r ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Leading dimensions for matrices
    int lda = (op_l == CUBLAS_OP_N) ? ni : nj; // lda depends on left matrix
    int ldb = (op_r == CUBLAS_OP_N) ? nj : nk; // ldb depends on right matrix
    int ldc = ni;                              // ldc is for output matrix

    // DOUT("lda: " << lda << ", ldb: " << ldb << ", ldc: " << ldc);

    // Scalars for the multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float **d_A_array = nullptr, **d_B_array = nullptr, **d_C_array = nullptr;

    handle_cuda_error(cudaMallocManaged(&d_A_array, batch_size * sizeof(float*)));
    handle_cuda_error(cudaMallocManaged(&d_B_array, batch_size * sizeof(float*)));
    handle_cuda_error(cudaMallocManaged(&d_C_array, batch_size * sizeof(float*)));

    for (int i = 0; i < batch_size; ++i) {
      d_A_array[i] = (float*)lhs + i * ni * nj;
      d_B_array[i] = (float*)rhs + i * nj * nk;
      d_C_array[i] = (float*)out + i * ni * nk;
    }

    std::vector<float*> h_A_array(batch_size, nullptr);

    // Set the cuBLAS stream
    cublasSetStream(cublas_handle, stream);

    // Perform batched matrix multiplication
    handle_cublas_error(cublasSgemmBatched(
        cublas_handle,
        op_l, op_r,
        ni, nk, nj,
        &alpha,
        d_A_array, lda,
        d_B_array, ldb,
        &beta,
        d_C_array, ldc,
        batch_size
    ));

    // debug: print the output matrix
    if (debug) {
      DOUT("Output matrix after computation: ");
      printFloatGPU(out, ni*nk*batch_size);
    }

    // cudaStreamSynchronize(stream);
}

contraction_stream_t
contraction_stream_t::make(
  dtype_t dtype,
  vector<uint64_t> const& shape,
  vector<int> const& lhs_inn_modes,
  vector<int> const& rhs_inn_modes,
  int out_rank)
{  
  if (dtype != dtype_t::f32) {
    throw std::runtime_error("Testing function; only f32 is supported");
  }
  auto dsz = dtype_size(dtype);
  auto maybe = contraction_stream_t::make_bs_is_js_ks(
    lhs_inn_modes, rhs_inn_modes, out_rank);
  if(!maybe) {
    throw std::runtime_error(
      "one-sided aggs like k in ijk,ij->i aren't supported "
      "nor are broadcasting outs like z in ij,jk->ikz");
  }

  auto& [bs,is,js,ks] = maybe.value();

  vector<uint64_t> lhs_shape;
  for(auto const& i: lhs_inn_modes) {
    lhs_shape.push_back(shape[i]);
  }
  vector<uint64_t> rhs_shape;
  for(auto const& i: rhs_inn_modes) {
    rhs_shape.push_back(shape[i]);
  }
  vector<uint64_t> out_shape(shape.begin(), shape.begin() + out_rank);
  vector<int> out_modes = vector_iota<int>(out_rank);

  uint64_t nb = 1; for(auto const& b: bs) { nb *= shape[b]; }
  uint64_t ni = 1; for(auto const& i: is) { ni *= shape[i]; }
  uint64_t nj = 1; for(auto const& j: js) { nj *= shape[j]; }
  uint64_t nk = 1; for(auto const& k: ks) { nk *= shape[k]; }

  optional<contraction_stream_t> ret;
  uint64_t best_cost;
  bool found_a_plan = false;
  for(bool lhs_t: {false, true}) {
  for(bool rhs_t: {false, true}) {
    contraction_stream_t::batching_t plan { bs, is, js, ks, lhs_t, rhs_t };

    auto perm_lhs =
      contraction_stream_t::permute_info_t::from_inn_shape(
        lhs_shape, lhs_inn_modes, plan.modes_lhs());
    auto perm_rhs =
      contraction_stream_t::permute_info_t::from_inn_shape(
        rhs_shape, rhs_inn_modes, plan.modes_rhs());
    auto perm_out =
      contraction_stream_t::permute_info_t::from_out_shape(
        out_shape, plan.modes_out(), out_modes);

    contraction_stream_t f {
      .dtype = dtype,
      .perm_lhs = std::nullopt,
      .perm_rhs = std::nullopt,
      .perm_out = std::nullopt,
      .nb = nb,
      .ni = ni,
      .nj = nj,
      .nk = nk,
      .stride_lhs = 0,
      .stride_rhs = 0,
      .stride_out = 0,
      .workspace_size = 0,
      .rhs_work_offset = 0,
      .out_work_offset = 0,
      .trans_lhs = lhs_t,
      .trans_rhs = rhs_t,
    };

    int nb_plan = plan.bs.size();
    int ni_plan = plan.is.size();

    int max_leading_lhsout = nb_plan + (plan.lhs_t ? 0 : ni_plan);
    int max_leading_rhs = nb_plan;

    int leading_lhs = std::min(max_leading_lhsout, perm_lhs.num_leading_modes());
    int leading_rhs = std::min(max_leading_rhs,    perm_rhs.num_leading_modes());
    int leading_out = std::min(max_leading_lhsout, perm_out.num_leading_modes());

    int leading_b = std::min(leading_lhs, std::min(leading_rhs, leading_out));
    int leading_i;
    if(leading_b == nb_plan) {
      leading_i = std::min(leading_lhs, leading_out) - leading_b;
    } else {
      leading_i = 0;
    }

    uint64_t cost = 0;
    uint64_t ws = 0;
    f.stride_lhs = dsz * product(perm_lhs.inn_shape) / (nb*ni);
    f.stride_rhs = dsz * product(perm_rhs.inn_shape) / nb;
    f.stride_out = dsz * product(perm_out.inn_shape) / (nb*ni);
    if (f.stride_lhs == 0 || f.stride_rhs == 0 || f.stride_out == 0) {
      throw std::runtime_error("Invalid strides");
    }

    if(!perm_lhs.is_no_op()) {
      cost += product(lhs_shape);
      // f.perm_lhs = permute_info_t {
      //   .inn_shape = perm_lhs.inn_shape,
      //   .out_perm  = perm_lhs.out_perm
      // };
      f.perm_lhs = perm_lhs.drop_leading(leading_b + leading_i);
      ws += dsz * product(f.perm_lhs.value().inn_shape);
    }
    f.rhs_work_offset = ws;
    if(!perm_rhs.is_no_op()) {
      cost += product(rhs_shape);
      // f.perm_rhs = permute_info_t {
      //   .inn_shape = perm_rhs.inn_shape,
      //   .out_perm  = perm_rhs.out_perm
      // };
      // ws += dsz * product(perm_rhs.inn_shape);
      f.perm_rhs = perm_rhs.drop_leading(leading_b);
      ws += dsz * product(f.perm_rhs.value().inn_shape);
    }
    f.out_work_offset = ws;
    if(!perm_out.is_no_op()) {
      cost += product(out_shape);
      // f.perm_out = permute_info_t {
      //   .inn_shape = perm_out.inn_shape,
      //   .out_perm  = perm_out.out_perm
      // };
      // ws += dsz * product(perm_out.inn_shape);
      f.perm_out = perm_out.drop_leading(leading_b + leading_i);
      ws += dsz * product(f.perm_out.value().inn_shape);
    }

    // setting other parameters
    f.workspace_size = ws;

    int64_t total_b_elem = 1;
    for(auto const& x: plan.bs) { total_b_elem *= shape[x]; }
    uint64_t total_i_elem = 1;
    for(auto const& x: plan.is) { total_i_elem *= shape[x]; }
    uint64_t total_j_elem = 1;
    for(auto const& x: plan.js) { total_j_elem *= shape[x]; }
    uint64_t total_k_elem = 1;
    for(auto const& x: plan.ks) { total_k_elem *= shape[x]; }

    f.inner.nb = total_b_elem / f.nb ;
    f.inner.ni = total_i_elem / f.ni ;
    f.inner.nj = total_j_elem          ;
    f.inner.nk = total_k_elem          ;

    f.inner.lhs_t = plan.lhs_t;
    f.inner.rhs_t = plan.rhs_t;


    if(ret) {
      if(cost < best_cost) {
        found_a_plan = true;
        best_cost = cost;
        ret = f;
      }
    } else {
      found_a_plan = true;
      best_cost = cost;
      ret = f;
    }
  }}

  if (!found_a_plan) {
    throw std::runtime_error("No valid plan found");
  }

  return ret.value();
}

void contraction_stream_t::print_info(){
  DOUT("nb: " << nb);
  DOUT("ni: " << ni);
  DOUT("nj: " << nj);
  DOUT("nk: " << nk);
  DOUT("stride_lhs: " << stride_lhs);
  DOUT("stride_rhs: " << stride_rhs);
  DOUT("stride_out: " << stride_out);
  DOUT("workspace_size: " << workspace_size);
  DOUT("lhs_work_offset: " << lhs_work_offset);
  DOUT("rhs_work_offset: " << rhs_work_offset);
  DOUT("out_work_offset: " << out_work_offset);
  DOUT("trans_lhs: " << trans_lhs);
  DOUT("trans_rhs: " << trans_rhs);
  DOUT("inner.nb: " << inner.nb);
  DOUT("inner.ni: " << inner.ni);
  DOUT("inner.nj: " << inner.nj);
  DOUT("inner.nk: " << inner.nk);
  DOUT("inner.lhs_t: " << inner.lhs_t);
  DOUT("inner.rhs_t: " << inner.rhs_t);

  if (perm_lhs) {
    DOUT("perm_lhs: ");
    DOUT("inn_shape: " << perm_lhs.value().inn_shape);
    DOUT("out_perm: " << perm_lhs.value().out_perm);
  }
  else {
    DOUT("perm_lhs: None");
  }

  if (perm_rhs) {
    DOUT("perm_rhs: ");
    DOUT("inn_shape: " << perm_rhs.value().inn_shape);
    DOUT("out_perm: " << perm_rhs.value().out_perm);
  }
  else {
    DOUT("perm_rhs: None");
  }

  if (perm_out) {
    DOUT("perm_out: ");
    DOUT("inn_shape: " << perm_out.value().inn_shape);
    DOUT("out_perm: " << perm_out.value().out_perm);
  }
  else {
    DOUT("perm_out: None");
  }
}

void contraction_stream_t::operator()(
  void* workspace,
  void* out,
  void const* lhs, void const* rhs,
  void* lhs_gpu, void* rhs_gpu,
  void* out_gpu,
  int batch_per_move) const
{
  // batch_per_move is the batch size we move to gpu at a time
  // assume nb / batch_per_move is an integer
  if (inner.nb > batch_per_move && inner.nb % batch_per_move != 0) {
    throw std::runtime_error("nb must be divisible by batch_per_move");
  }
  uint64_t dsize = dtype_size(dtype);

  uint64_t num_moves = inner.nb / batch_per_move;

  void* lhs_work_ = (char*)workspace + lhs_work_offset;
  void* rhs_work_ = (char*)workspace + rhs_work_offset;
  void* out_work_ = (char*)workspace + out_work_offset;

  // create handle
  cublasHandle_t cublas_handle;
  handle_cublas_error(cublasCreate(&cublas_handle), "cublasCreate");
  // create 2 streams: one for moving and one for computation
  cudaStream_t moving_stream;
  handle_cuda_error(cudaStreamCreate(&moving_stream));
  cudaStream_t computation_stream;
  handle_cuda_error(cudaStreamCreate(&computation_stream));

  for(uint64_t b = 0; b != nb; ++b) {
    for(uint64_t i = 0; i != ni; ++i) {
      void* out_data = (char*)out + stride_out*(b*ni + i);
      void* lhs_data = (char*)lhs + stride_lhs*(b*ni + i);
      void* rhs_data = (char*)rhs + stride_rhs*b;

      void* lhs_work;
      if(perm_lhs) {
        lhs_work = lhs_work_;
        permute_t lhs_perm(1024);
        lhs_perm(perm_lhs.value().inn_shape,
          perm_lhs.value().out_perm,
          (float*)lhs_work, (float*)lhs_data);
      } else {
        lhs_work = lhs_data;
      }

      void* rhs_work;
      // DOUT("rhs_data: ");
      // printFloatCPU((float*)rhs_data, 4);
      if(perm_rhs) {
        rhs_work = rhs_work_;
        permute_t rhs_perm(1024);
        rhs_perm(perm_rhs.value().inn_shape,
          perm_rhs.value().out_perm,
          (float*)rhs_work, (float*)rhs_data);
      } else {
        rhs_work = rhs_data;
      }
      // DOUT("rhs_work: ");
      // printFloatCPU((float*)rhs_work, 4);

      void* out_work;
      if(perm_out) {
        out_work = out_work_;
      } else {
        out_work = out_data;
      }

      cudaEvent_t event_move_inputs;
      cudaEvent_t event_compute;
      cudaEvent_t event_move_output;

      handle_cuda_error(cudaEventCreate(&event_compute), "cudaEventCreate");
      handle_cuda_error(cudaEventCreate(&event_move_output), "cudaEventCreate");
      handle_cuda_error(cudaEventCreate(&event_move_inputs), "cudaEventCreate");

      uint64_t batch_size = batch_per_move;
      if (inner.nb > batch_per_move) {
        num_moves = 1;
        batch_size = inner.nb;
      }

      for (uint64_t b_move = 0; b_move != num_moves; ++b_move){
        // DOUT("b_move: " << b_move);
        // using double buffering, so each gpu pointer should fit 2 batches
        bool left = b_move % 2 == 0;
        auto current_lhs_gpu = increment_void_ptr(lhs_gpu, left ? 0 : batch_size * inner.ni * inner.nj * dsize);
        auto current_rhs_gpu = increment_void_ptr(rhs_gpu, left == 0 ? 0 : batch_size * inner.nj * inner.nk * dsize);
        auto current_out_gpu = increment_void_ptr(out_gpu, left == 0 ? 0 : batch_size * inner.ni * inner.nk * dsize);

        auto lhs_cpu = increment_void_ptr(lhs_work, b_move * dsize*inner.ni*inner.nj);
        auto rhs_cpu = increment_void_ptr(rhs_work, b_move * dsize*inner.nj*inner.nk);

        // DOUT("lhs_cpu: ");
        // printFloatCPU((float*)lhs_cpu, ni*nj);
        // DOUT("rhs_cpu: ");
        // printFloatCPU((float*)rhs_cpu, nj*nk);
        
        // if (b_move > 0){
        //   handle_cuda_error(cudaStreamWaitEvent(moving_stream, event_move_output));
        // }
        handle_cuda_error(cudaMemcpyAsync(current_lhs_gpu, lhs_cpu, 
                                          batch_size * inner.ni * inner.nj * dsize, 
                                          cudaMemcpyHostToDevice, moving_stream));
        handle_cuda_error(cudaMemcpyAsync(current_rhs_gpu, rhs_cpu, 
                                          batch_size * inner.nj * inner.nk * dsize,
                                          cudaMemcpyHostToDevice, moving_stream));

        // cudaStreamSynchronize(moving_stream);

        handle_cuda_error(cudaEventRecord(event_move_inputs, moving_stream));

        // compute the matrix multiplication on the computation stream
        handle_cuda_error(cudaStreamWaitEvent(computation_stream, event_move_inputs));
        
        batch_matmul_gpu(dtype, inner.ni, inner.nj, inner.nk, inner.lhs_t, inner.rhs_t, 
          current_out_gpu, current_lhs_gpu, current_rhs_gpu, computation_stream, cublas_handle, batch_size);

        handle_cuda_error(cudaEventRecord(event_compute, computation_stream));

        handle_cuda_error(cudaStreamWaitEvent(moving_stream, event_compute));

        // copy the result back to the output
        handle_cuda_error(cudaMemcpyAsync(out_work, current_out_gpu, batch_size * inner.ni * inner.nk * dsize, 
                                          cudaMemcpyDeviceToHost, moving_stream));
                                        
        cudaEventRecord(event_move_output, moving_stream);
        // cudaStreamSynchronize(moving_stream);
        out_work = increment_void_ptr(out_work, dsize*inner.ni*inner.nk);
      }

      cudaStreamSynchronize(moving_stream);

      // check if we need to permute the output on the CPU
      if (perm_out) {
        permute_t out_perm(1024);
        out_perm(perm_out.value().inn_shape,
          perm_out.value().out_perm,
          (float*)out_data, (float*)out_work);
      }
    }
  }
}

// we assume that all inputs will be in the same gpu
void streaming_matmul(int ni, int nj, int nk, int num_workspace_buffers, 
                      int num_partitions) {
  // first check if we can paritition the input tensors nicely (paritiioning on nj)
  if (nj % num_partitions != 0) {
    throw std::runtime_error("nj must be divisible by num_partitions");
  }

  // initiate the tensor on the cpu first
  dtype_t dtype = dtype_t::f32;
  dbuffer_t lhs = make_dbuffer(dtype, ni * nj);
  dbuffer_t rhs = make_dbuffer(dtype, nj * nk);
  // random initializing the tensors for lhs and rhs
  lhs.random();
  dbuffer_t out_cpu = make_dbuffer(dtype, ni * nk);
  out_cpu.zeros();

  // allocate the buffers on gpu
  // workspace buffer will store the intermediate results
  vector<void*> workspace_buffers;
  vector<void*> lhs_partitions;
  vector<void*> rhs_partitions;
  for (int i = 0; i < num_workspace_buffers; i++) {
    void* workspace;
    cudaMalloc(&workspace, ni * nj * sizeof(float));
    // each workspace buffer is of size ni * nj
    workspace_buffers.push_back(workspace);
    void* lhs_partition;
    // each lhs partition is of size ni * nj/num_partitions
    int j_partition = nj/num_partitions;
    handle_cuda_error(cudaMalloc(&lhs_partition, ni * j_partition * sizeof(float)));
    lhs_partitions.push_back(lhs_partition);
    void* rhs_partition;
    // each rhs partition is of size nj/num_partitions * nk
    handle_cuda_error(cudaMalloc(&rhs_partition, j_partition * nk * sizeof(float)));
    rhs_partitions.push_back(rhs_partition);
  }

  // we assume that the GPU will have space for the output
  void* out;
  cudaMalloc(&out, ni * nk * sizeof(float));
  // copy out_cpu to out
  cudaMemcpy(out, out_cpu.ptr(), ni * nk * sizeof(float), cudaMemcpyHostToDevice);

  // every lhs partition will be the shape of ni x nj/num_partitions
  // every rhs partition will be the shape of nj/num_partitions x nk

  // for every partial output, the GPU need to do the following: (attached to a stream)
  // 1. copy the lhs partition to the lhs_partitions 
  // 2. copy the rhs partition to the rhs_partitions
  // 3. do the matrix multiplication
  // 4. copy the result to the corresponding workspace buffer
  // every time we fill a workspace buffer, we can aggregate the result to the output buffer

  // start the timer
  auto start = std::chrono::high_resolution_clock::now();

  cublasHandle_t cublas_handle;
  handle_cublas_error(cublasCreate(&cublas_handle), "cublasCreate"); 

  // there would be 2 streams: one for the moving and one for the computation
  cudaStream_t moving_stream;
  handle_cuda_error(cudaStreamCreate(&moving_stream));
  cudaStream_t computation_stream;
  handle_cuda_error(cudaStreamCreate(&computation_stream));

  for (int i = 0; i < num_partitions; i++) {
    DOUT("Partition " << i << " out of " << num_partitions);
    auto current_stream = moving_stream;
    auto current_lhs = lhs_partitions[i % num_workspace_buffers];
    auto current_rhs = rhs_partitions[i % num_workspace_buffers];
    auto current_workspace = workspace_buffers[i % num_workspace_buffers];
    // create the cuda event
    cudaEvent_t event_move;
    handle_cuda_error(cudaEventCreate(&event_move), "cudaEventCreate");
    // copy the lhs partition to the lhs_partitions
    auto lhs_cpu = increment_void_ptr(lhs.ptr(), i * ni * nj/num_partitions * sizeof(float));
    auto rhs_cpu = increment_void_ptr(rhs.ptr(), i * nj/num_partitions * nk * sizeof(float));
    handle_cuda_error(cudaMemcpyAsync(current_lhs, lhs_cpu, ni * nj/num_partitions * sizeof(float), 
                                      cudaMemcpyHostToDevice, current_stream));
    // copy the rhs partition to the rhs_partitions
    handle_cuda_error(cudaMemcpyAsync(current_rhs, rhs_cpu, nj/num_partitions * nk * sizeof(float), 
                                      cudaMemcpyHostToDevice, current_stream));
    // record the event
    handle_cuda_error(cudaEventRecord(event_move, current_stream));

    // do the matrix multiplication
    
    // wait for the moving stream to finish
    handle_cuda_error(cudaStreamWaitEvent(computation_stream, event_move));
    handle_cublas_error(cublasSetStream(cublas_handle, computation_stream), "cublasSetStream");
    float alpha = 1.0;
    handle_cublas_error(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, ni, nk, nj/num_partitions, 
                                    &alpha, static_cast<float const*>(current_lhs), ni, 
                                    static_cast<float const*>(current_rhs), nj/num_partitions, 
                                    &alpha, static_cast<float*>(out), ni), 
                                    "cublasSgemm");
  }
  // in the end, we need to wait for the computation stream to finish
  cudaDeviceSynchronize();
}

// Check CUDA calls for errors
#define CUDA_CALL(func)                                                        \
    {                                                                          \
        cudaError_t err = (func);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__  \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Check cuBLAS calls for errors
#define CUBLAS_CALL(func)                                                      \
    {                                                                          \
        cublasStatus_t status = (func);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error in " << __FILE__ << " line " << __LINE__\
                      << std::endl;                                            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

void batch_gemm_test(){
  const int M = 2; // Rows of A and C
  const int N = 2; // Columns of B and C
  const int K = 2; // Columns of A and rows of B
  const int batchCount = 2; // Number of matrices in the batch

  // Host data (row-major order)
  float h_A[batchCount][M * K] = { {1, 2, 3, 4}, {5, 6, 7, 8} }; // Batch of A
  float h_B[batchCount][K * N] = { {1, 0, 0, 1}, {1, 1, 1, 1} }; // Batch of B
  float h_C[batchCount][M * N] = { 0 };                          // Batch of C

  // Device pointers for each matrix
  float *d_A[batchCount], *d_B[batchCount], *d_C[batchCount];

  // Allocate device memory for each matrix in the batch
  for (int i = 0; i < batchCount; i++) {
      CUDA_CALL(cudaMalloc((void**)&d_A[i], M * K * sizeof(float)));
      CUDA_CALL(cudaMalloc((void**)&d_B[i], K * N * sizeof(float)));
      CUDA_CALL(cudaMalloc((void**)&d_C[i], M * N * sizeof(float)));
  }

  // Copy data from host to device
  for (int i = 0; i < batchCount; i++) {
      CUDA_CALL(cudaMemcpy(d_A[i], h_A[i], M * K * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpy(d_B[i], h_B[i], K * N * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Create cuBLAS handle
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));

  // Attach the stream to the cuBLAS handle
  CUBLAS_CALL(cublasSetStream(handle, stream));

  // Create device arrays of pointers
  float **d_A_array, **d_B_array, **d_C_array;
  CUDA_CALL(cudaMalloc((void**)&d_A_array, batchCount * sizeof(float*)));
  CUDA_CALL(cudaMalloc((void**)&d_B_array, batchCount * sizeof(float*)));
  CUDA_CALL(cudaMalloc((void**)&d_C_array, batchCount * sizeof(float*)));

  CUDA_CALL(cudaMemcpy(d_A_array, d_A, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B_array, d_B, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_C_array, d_C, batchCount * sizeof(float*), cudaMemcpyHostToDevice));

  // Scalars
  float alpha = 1.0f;
  float beta = 0.0f;

  // Perform batched matrix multiplication
  CUBLAS_CALL(cublasSgemmBatched(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
      M, N, K,                  // Dimensions
      &alpha,                   // Alpha
      (const float**)d_A_array, M,  // A and its leading dimension
      (const float**)d_B_array, K,  // B and its leading dimension
      &beta,                    // Beta
      d_C_array, M,             // C and its leading dimension
      batchCount                // Number of matrices in the batch
  ));

  // Wait for stream to finish
  CUDA_CALL(cudaStreamSynchronize(stream));

  // Copy result back to host
  for (int i = 0; i < batchCount; i++) {
      CUDA_CALL(cudaMemcpy(h_C[i], d_C[i], M * N * sizeof(float), cudaMemcpyDeviceToHost));
  }

  // Print result
  std::cout << "Resulting matrices C:\n";
  for (int b = 0; b < batchCount; b++) {
      std::cout << "Matrix " << b << ":\n";
      for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
              std::cout << h_C[b][i * N + j] << " ";
          }
          std::cout << "\n";
      }
  }

  // Cleanup
  for (int i = 0; i < batchCount; i++) {
      CUDA_CALL(cudaFree(d_A[i]));
      CUDA_CALL(cudaFree(d_B[i]));
      CUDA_CALL(cudaFree(d_C[i]));
  }
  CUDA_CALL(cudaFree(d_A_array));
  CUDA_CALL(cudaFree(d_B_array));
  CUDA_CALL(cudaFree(d_C_array));

  CUBLAS_CALL(cublasDestroy(handle));

  // Destroy the stream
  CUDA_CALL(cudaStreamDestroy(stream));
}

void sgemm_test(){
  int ni = 1;
  int nj = 2;
  int nk = 2;
  bool trans_l = false;
  bool trans_r = false;
  cudaSetDevice(0);
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));
  uint64_t num_batches = 1;
  float* out;
  float* lhs;
  float* rhs;
  CUDA_CALL(cudaMalloc(&out, sizeof(float) * num_batches * ni * nk));
  CUDA_CALL(cudaMalloc(&lhs, sizeof(float) * num_batches * ni * nj));
  CUDA_CALL(cudaMalloc(&rhs, sizeof(float) * num_batches * nj * nk));
  // fill the input tensors with 1s
  float* lhs_cpu = new float[num_batches * ni * nj];
  float* rhs_cpu = new float[num_batches * nj * nk];
  for (int i = 0; i < num_batches * ni * nj; i++) {
    lhs_cpu[i] = 1.0;
  }
  for (int i = 0; i < num_batches * nj * nk; i++) {
    rhs_cpu[i] = 1.0;
  }
  float* out_cpu = new float[num_batches * ni * nk];
  for (int i = 0; i < num_batches * ni * nk; i++) {
    out_cpu[i] = 0.0;
  }
  CUDA_CALL(cudaMemcpy(lhs, lhs_cpu, sizeof(float) * num_batches * ni * nj, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(rhs, rhs_cpu, sizeof(float) * num_batches * nj * nk, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(out, out_cpu, sizeof(float) * num_batches * ni * nk, cudaMemcpyHostToDevice));
  
  batch_matmul_gpu(dtype_t::f32, ni, nj, nk, trans_l, trans_r, out, lhs, rhs, stream, handle, num_batches);

  cudaStreamSynchronize(stream);
  cudaSetDevice(0);
  cudaDeviceSynchronize();

  // bring the result back to the host
  float* out_cpu_result = new float[num_batches * ni * nk];
  CUDA_CALL(cudaMemcpy(out_cpu_result, out, sizeof(float) * num_batches * ni * nk, cudaMemcpyDeviceToHost));
  DOUT("Output matrix after computation: ");
  printFloatCPU(out_cpu_result, num_batches * ni * nk);
}

void streaming_test(){
    bool debug = false;
    dtype_t dtype = dtype_t::f32;

    // einsummable_t e(
    // {1, 2, 2, 2, 2},
    // { {0,1,2,4}, {0,4,1,3} },
    // 4,
    // scalarop_t::make_mul(dtype),
    // castable_t::add);

    einsummable_t e(
    {20, 40, 40, 40, 40},
    { {0,1,2,4}, {0,4,1,3} },
    4,
    scalarop_t::make_mul(dtype),
    castable_t::add);

    auto inn_shapes = e.inn_shapes();
    dbuffer_t lhs = make_dbuffer(dtype, product(inn_shapes[0]));
    dbuffer_t rhs = make_dbuffer(dtype, product(inn_shapes[1]));

    lhs.ones();
    rhs.ones();

    if (debug){
      DOUT("lhs: ");
      printFloatCPU((float*)lhs.ptr(), lhs.size() / sizeof(float));
      DOUT("rhs: ");
      printFloatCPU((float*)rhs.ptr(), rhs.size() / sizeof(float));

      dbuffer_t out_ref = reference_einsummable(e, {lhs, rhs});
      DOUT("reference output: ");
      printFloatCPU((float*)out_ref.ptr(), out_ref.size() / sizeof(float));
      DOUT("reference output sum: ");
      DOUT(out_ref.sum_to_f64());
    }

    dbuffer_t out = make_dbuffer(dtype, e.out_nelem());
    out.fill(scalar_t(1.0f));

    auto num_floats = out.size() / sizeof(float);

    DOUT("num of floats: " << num_floats);

    contraction_stream_t c = contraction_stream_t::make(
      dtype_t::f32,
      e.join_shape,
      e.inns[0], e.inns[1],
      e.out_rank
    );

    c.print_info();

    dbuffer_t workspace = make_dbuffer(dtype, c.workspace_size);

    // pin the CPU memory on the host for faster transfer
    cudaHostRegister(workspace.ptr(), workspace.size(), cudaHostRegisterDefault);
    cudaHostRegister(out.ptr(), out.size(), cudaHostRegisterDefault);
    cudaHostRegister(lhs.ptr(), lhs.size(), cudaHostRegisterDefault);
    cudaHostRegister(rhs.ptr(), rhs.size(), cudaHostRegisterDefault);

    // double buffer so the memory pointer on GPU should be double the size of the size of a single block
    void* lhs_gpu;
    void* rhs_gpu;
    void* out_gpu;
    
    uint64_t batch_per_move = 1;

    cudaMalloc(&lhs_gpu, 2 * sizeof(float) * batch_per_move * c.ni * c.nj);
    cudaMalloc(&rhs_gpu, 2 * sizeof(float) * batch_per_move * c.nj * c.nk);
    cudaMalloc(&out_gpu, 2 * sizeof(float) * batch_per_move * c.ni * c.nk);
    cudaProfilerStart();
    c(workspace.ptr(), out.ptr(), lhs.ptr(), rhs.ptr(), lhs_gpu, rhs_gpu, out_gpu, batch_per_move);
    cudaProfilerStop();
    cudaDeviceSynchronize();

    if (debug){
      DOUT("output: ");
      printFloatCPU((float*)out.ptr(), num_floats);

      DOUT("output sum: ");
      DOUT(out.sum_to_f64());
    }
}
void error_example(){
  einsummable_t e(
    {1, 2, 2, 2, 2},
    { {0,1,2,4}, {0,4,1,3} },
    4,
    scalarop_t::make_mul(dtype_t::f32),
    castable_t::add);

  auto dsz = dtype_size(dtype_t::f32);
  auto lhs_inn_modes = e.inns[0];
  auto rhs_inn_modes = e.inns[1];
  auto shape = e.join_shape;
  auto out_rank = e.out_rank;

  auto maybe = contraction_stream_t::make_bs_is_js_ks(
    lhs_inn_modes, rhs_inn_modes, out_rank);
  if(!maybe) {
    throw std::runtime_error(
      "one-sided aggs like k in ijk,ij->i aren't supported "
      "nor are broadcasting outs like z in ij,jk->ikz");
  }

  auto& [bs,is,js,ks] = maybe.value();

  vector<uint64_t> lhs_shape;
  for(auto const& i: lhs_inn_modes) {
    lhs_shape.push_back(shape[i]);
  }
  vector<uint64_t> rhs_shape;
  for(auto const& i: rhs_inn_modes) {
    rhs_shape.push_back(shape[i]);
  }
  vector<uint64_t> out_shape(shape.begin(), shape.begin() + out_rank);
  vector<int> out_modes = vector_iota<int>(out_rank);

  uint64_t nb = 1; for(auto const& b: bs) { nb *= shape[b]; }
  uint64_t ni = 1; for(auto const& i: is) { ni *= shape[i]; }
  uint64_t nj = 1; for(auto const& j: js) { nj *= shape[j]; }
  uint64_t nk = 1; for(auto const& k: ks) { nk *= shape[k]; }

  DOUT("nb: " << nb);
  DOUT("ni: " << ni);
  DOUT("nj: " << nj);
  DOUT("nk: " << nk);

  struct plan_t {
    // bij,bjk->bik
    // bij,bkj->bik
    // bji,bjk->bik
    // bji,bkj->bik
    vector<int> bs;
    vector<int> is;
    vector<int> js;
    vector<int> ks;
    bool lhs_t;
    bool rhs_t;

    vector<int> modes_lhs() const {
      return vector_concatenate(bs,
        lhs_t                      ?
        vector_concatenate(js, is) :
        vector_concatenate(is, js));
    }
    vector<int> modes_rhs() const {
      return vector_concatenate(bs,
        rhs_t                      ?
        vector_concatenate(ks, js) :
        vector_concatenate(js, ks));
    }
    vector<int> modes_out() const {
      return vector_concatenate(bs, vector_concatenate(is, ks));
    }
  };

  plan_t plan { bs, is, js, ks, false, true };

  auto perm_lhs =
    contraction_stream_t::permute_info_t::from_inn_shape(
      lhs_shape, lhs_inn_modes, plan.modes_lhs());
  auto perm_rhs =
    contraction_stream_t::permute_info_t::from_inn_shape(
      rhs_shape, rhs_inn_modes, plan.modes_rhs());
  auto perm_out =
    contraction_stream_t::permute_info_t::from_out_shape(
      out_shape, plan.modes_out(), out_modes);

  // if (!perm_lhs.is_no_op){
    DOUT("perm_lhs: ");
    DOUT("inn_shape: " << perm_lhs.inn_shape);
    DOUT("out_perm: " << perm_lhs.out_perm);
  // }
  // if (!perm_rhs.is_no_op){
    DOUT("perm_rhs: ");
    DOUT("inn_shape: " << perm_rhs.inn_shape);
    DOUT("out_perm: " << perm_rhs.out_perm);
  // }
  // if (!perm_out.is_no_op){
    DOUT("perm_out: ");
    DOUT("inn_shape: " << perm_out.inn_shape);
    DOUT("out_perm: " << perm_out.out_perm);
  // }

  auto stride_lhs = dsz * product(perm_lhs.inn_shape) / (nb*ni);
  auto stride_rhs = dsz * product(perm_rhs.inn_shape) / nb;
  auto stride_out = dsz * product(perm_out.inn_shape) / (nb*ni);

  DOUT("stride_lhs: " << stride_lhs);
  DOUT("stride_rhs: " << stride_rhs);
  DOUT("stride_out: " << stride_out);
}

int main(int argc, char** argv) {
  // {
  //   if (argc != 6) {
  //     std::cerr << "Usage: ./stream_matmul ni nj nk num_workspace_buffers num_partitions" << std::endl;
  //     return 1;
  //   }
  //   int ni = std::stoi(argv[1]);
  //   int nj = std::stoi(argv[2]);
  //   int nk = std::stoi(argv[3]);
  //   int num_workspace_buffers = std::stoi(argv[4]);
  //   int num_partitions = std::stoi(argv[5]);
  //   streaming_matmul(ni, nj, nk, num_workspace_buffers, num_partitions);
  // }
  
  {
    streaming_test();
  }

  {
    // sgemm_test();
    // batch_gemm_test();
  }  
}
