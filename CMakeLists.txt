cmake_minimum_required(VERSION 3.2)

project(einsummable)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-sign-compare")
#set(CMAKE_CXX_FLAGS "-Wmaybe-uninitialized")
# set(CMAKE_CXX_FLAGS_RELEASE "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -pg -fno-omit-frame-pointer -DNDEBUG")#-fno-inline-functions -fno-inline-functions-called-once -fno-optimize-sibling-calls")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# TODO: make sure protocol buffers are built in the correct order
#       with respect to other files. Sometimes make -j is required
#       when make fails
# Note: running make proto will build proto

find_package(Protobuf REQUIRED)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/proto)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/half-2.2.0/include)
include_directories(${Protobuf_INCLUDE_DIRS})

add_subdirectory(src/proto)

file(GLOB BA_FILES ${PROJECT_SOURCE_DIR}/src/base/*.cc)
file(GLOB ES_FILES ${PROJECT_SOURCE_DIR}/src/einsummable/*.cc)
file(GLOB MG_FILES ${PROJECT_SOURCE_DIR}/src/matrixgraph/*.cc)
file(GLOB AP_FILES ${PROJECT_SOURCE_DIR}/src/autoplace/*.cc)
file(GLOB MS_FILES ${PROJECT_SOURCE_DIR}/src/misc/*.cc)

add_library(einsummable
  STATIC
  ${BA_FILES}
  ${ES_FILES}
  ${MG_FILES}
  ${AP_FILES}
  ${MS_FILES}
)

function(add_base_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)
  target_link_libraries(${exec} einsummable proto ${Protobuf_LIBRARIES})
endfunction()

#add_base_application("make_3dmatmul"    "exps" "make_3dmatmul")
# add_base_application("reference"        "exps" "reference")        # TODO: delete and turn into tests
# add_base_application("copyregion"       "exps" "copyregion")       # TODO: delete and turn into tests
# add_base_application("scalarop"         "exps" "scalarop")         # TODO: delete and turn into tests
# add_base_application("allocator_test"   "exps" "allocator_test")   # TODO: delete and turn into tests
# add_base_application("einsummable_test" "exps" "einsummable_test") # TODO: delete and turn into tests
# add_base_application("gwise"            "exps" "gwise")            # TODO: delete and turn into tests
# add_base_application("autodiff_test"    "exps" "autodiff_test")    # TODO: delete and turn into tests

add_executable(tests_base tests/base.cc tests/test.cc)
target_link_libraries(tests_base einsummable proto ${Protobuf_LIBRARIES})

function(add_llama_application exec filename)
  add_executable(${exec} llama/${filename}.cc
    llama/modules.cc llama/misc.cc llama/builder.cc)
  target_link_libraries(${exec} einsummable proto ${Protobuf_LIBRARIES})
endfunction()

add_llama_application("exp_apart03" "exp_apart03")
add_llama_application("tg_order"    "tg_order")


#add_llama_application("llama_main"  "main")
#add_llama_application("llama_exp01" "exp01")
#add_llama_application("llama_exp02" "exp02")
#add_llama_application("llama_exp03" "exp03")
#add_llama_application("llama_autodiff_test" "autodiff_test")
add_llama_application("cpoint"               "cpoint")

if(${CPU_EXECUTION_ENGINE})
  if(NOT MKL_THREADING)
    set(MKL_THREADING sequential)
  endif()

  add_compile_definitions(CPU_EXEC)

  # Tell MKL to use the best instructions; if
  # this instruction set it goes to the next best
  # and so on...
  # (MKL is bad at finding the right instruction set and
  #  this helps empirically)
  set(MKL_ENABLE_INSTRUCTIONS AVX512_E4)

  find_package(MKL REQUIRED)


  if(NOT DEFINED UCX_PATH)
    message( SEND_ERROR "Pleased define UCX_PATH")
  endif()
  set(UCX_DIR "${UCX_PATH}/lib/cmake/ucx")
  include_directories(${UCX_PATH}/include)
  find_package(UCX REQUIRED COMPONENTS ucp uct ucs)

  include_directories(${MKL_INCLUDE})

  file(GLOB EG_FILES ${PROJECT_SOURCE_DIR}/src/engine/*.cc)
  file(GLOB SE_FILES ${PROJECT_SOURCE_DIR}/src/server/*.cc)
  file(GLOB CP_FILES ${PROJECT_SOURCE_DIR}/src/engine/cpu/*.cc)
  file(GLOB CPMG_FILES ${PROJECT_SOURCE_DIR}/src/engine/cpu/mg/*.cc)
  file(GLOB CPTG_FILES ${PROJECT_SOURCE_DIR}/src/engine/cpu/tg/*.cc)
  file(GLOB CS_FILES ${PROJECT_SOURCE_DIR}/src/server/cpu/*.cc)

  # TODO: how do you make it so that not all the ES_FILES and MG_FILES
  #       have to be rebuilt?
  add_library(cpu_execution_engine
    STATIC
    ${BA_FILES}
    ${ES_FILES}
    ${MG_FILES}
    ${AP_FILES}
    ${MS_FILES}
    ${EG_FILES}
    ${SE_FILES}
    ${CP_FILES}
    ${CPMG_FILES}
    ${CPTG_FILES}
    ${CS_FILES}
  )

  function(add_cpu_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)
  target_link_libraries(${exec}
    cpu_execution_engine
    MKL::MKL
    ucx::ucp ucx::ucs
    #${SENTENCEPIECE_DIR}/lib/libsentencepiece.so
    #${SENTENCEPIECE_DIR}/lib/libsentencepiece_train.so
    ${Protobuf_LIBRARIES}
    proto)
  endfunction()

  #add_cpu_application("cpuexec"               "exps" "cpuexec")
  #add_cpu_application("cpuexec_3d"            "exps" "cpuexec_3d")
  #add_cpu_application("cpuexec_mm"            "exps" "cpuexec_mm")
  #add_cpu_application("cpuexec_touch"         "exps" "cpuexec_touch")
  #add_cpu_application("cpuexec_permute"       "exps" "cpuexec_permute")
  add_cpu_application("cpuexec_kernels"       "exps" "cpuexec_kernels")
  add_cpu_application("cpuexec_gen"           "exps" "cpuexec_gen")
  #add_cpu_application("cpumanager"            "exps" "cpumanager")
  #add_cpu_application("cpukerneltime"         "exps" "cpukerneltime")
  #add_cpu_application("cpumms"                "exps" "cpumms")
  #add_cpu_application("cpu_engine"            "exps" "cpu_engine")
  #add_cpu_application("cpu_engine_ml"         "exps" "cpu_engine_ml")
  #add_cpu_application("cpu_engine_server"     "exps" "cpu_engine_server")
  #add_cpu_application("cpu_engine_lff"        "exps" "cpu_engine_lff")
  #add_cpu_application("cpu_comm"              "exps" "cpu_comm")
  #add_cpu_application("eventloop"             "exps" "eventloop")
  #add_cpu_application("cpu_engine_run"        "exps" "cpu_engine_run")
  #add_cpu_application("ff_autodiff"           "exps" "ff_autodiff")
  #add_cpu_application("senexp"                "exps" "senexp")
  add_cpu_application("xtreme"                "exps" "xtreme")
  add_cpu_application("abc"                   "exps" "abc")
  add_cpu_application("cpuexeccompare"        "exps" "cpuexeccompare")

  function(add_cpu_llama_application exec filename)
    add_executable(${exec} llama/${filename}.cc
      llama/modules.cc llama/builder.cc llama/misc.cc llama/reader.cc
      llama/dataset_reader.cc llama/piper.cc)
    target_link_libraries(${exec}
      cpu_execution_engine
      MKL::MKL
      ucx::ucp ucx::ucs
      ${Protobuf_LIBRARIES}
      proto)
  endfunction()

  #add_cpu_llama_application("llama_cpu" "cpu2")
  #add_cpu_llama_application("inference_exp" "inference_exp")
  add_cpu_llama_application("llama_first_token" "first_token")

  # For some reason sentencepiece + protocolbuffers + this application
  # isn't cooperating well. Something to do with symbols + protocol
  # buffers being used in sentence piece and this application. So instead
  # of using sentence piece directly, train1 launches the tokenizer
  # executable and reads and writes from it.
  # And the tokenizer executable calls sentencepiece.

  #add_cpu_llama_application("train1" "train1")
  add_cpu_llama_application("train2" "train2")
  add_cpu_llama_application("trainsize" "trainsize")

endif()

if(${SENTENCEPIECE_EXP})
  message(STATUS ${SENTENCEPIECE_DIR})
  include_directories(${SENTENCEPIECE_DIR}/include)

  function(add_sp_application exec dir filename)
  add_executable(${exec} ${dir}/${filename}.cc)
  target_link_libraries(${exec}
    ${SENTENCEPIECE_DIR}/lib/libsentencepiece.so
    ${SENTENCEPIECE_DIR}/lib/libsentencepiece_train.so)
  endfunction()
  add_sp_application("llama_tokenizer" "llama" "tokenizer")

endif()

if(${GPU_EXECUTION_ENGINE})
  enable_language(CUDA)

  find_package(CUDA 12.6 REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")
  include_directories(${CUTENSOR_ROOT}/include)
  include_directories(${PROJECT_SOURCE_DIR}/src/engine)
  include_directories(${PROJECT_SOURCE_DIR}/src/engine/gpu)
  include_directories(${PROJECT_SOURCE_DIR}/src/server)
  include_directories(${PROJECT_SOURCE_DIR}/src/server/gpu)

  add_compile_definitions(GPU_EXEC)

  # add ucx dependency
  # define UCX_PATH below
  if(NOT DEFINED UCX_PATH)
    message( SEND_ERROR "Pleased define UCX_PATH")
  endif()
  set(UCX_DIR "${UCX_PATH}/lib/cmake/ucx")
  include_directories(${UCX_PATH}/include)
  find_package(UCX REQUIRED COMPONENTS ucp uct ucs)

  # set the right architecture
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

  # file(GLOB CP_FILES ${PROJECT_SOURCE_DIR}/src/execution/gpu/*.cc)
  file(GLOB CU_FILES ${PROJECT_SOURCE_DIR}/src/engine/gpu/*.cu)
  file(GLOB EG_FILES ${PROJECT_SOURCE_DIR}/src/engine/*.cc)
  file(GLOB GP_FILES ${PROJECT_SOURCE_DIR}/src/engine/gpu/*.cc)
  file(GLOB SE_FILES ${PROJECT_SOURCE_DIR}/src/server/*.cc)
  file(GLOB SS_FILES ${PROJECT_SOURCE_DIR}/src/server/gpu/*.cc)

  add_library(cutensor SHARED IMPORTED)
  set_target_properties(cutensor PROPERTIES
    IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/include")

  add_library(gpu_execution_engine
    STATIC
    ${BA_FILES}
    ${ES_FILES}
    ${MG_FILES}
    ${AP_FILES}
    ${MS_FILES}
    ${CU_FILES}
    ${EG_FILES}
    ${GP_FILES}
    ${SE_FILES}
    ${SS_FILES}
  )

  function(add_gpu_llama_application exec filename)
    add_executable(${exec} llama/${filename}.cc
      llama/modules.cc llama/builder.cc llama/misc.cc llama/reader.cc
      llama/dataset_reader.cc llama/piper.cc)
    target_link_libraries(${exec}
      gpu_execution_engine
      proto
      ${Protobuf_LIBRARIES}
      ${CUDA_LIBRARIES}
      cutensor
      /usr/local/cuda-12.6/lib64/libcudart.so
      /usr/local/cuda-12.6/lib64/libcublasLt.so
      /usr/local/cuda-12.6/lib64/libcublas.so
      ucx::ucp ucx::ucs)
  endfunction()

  ## target_link_libraries(${exec}
  ##   gpu_execution_engine
  ##   proto
  ##   ${Protobuf_LIBRARIES}
  ##   ${CUDA_LIBRARIES}
  ##   cutensor
  ##   cudart
  ##   cublasLt
  ##   cublas
  ##   ucx::ucp ucx::ucs)
  #target_link_libraries(${exec}
  #  gpu_execution_engine
  #  proto
  #  ${Protobuf_LIBRARIES}
  #  ${CUDA_LIBRARIES}
  #  cutensor
  #  /usr/local/cuda-12.6/lib64/libcudart.so
  #  /usr/local/cuda-12.6/lib64/libcublasLt.so
  #  /usr/local/cuda-12.6/lib64/libcublas.so
  #  ucx::ucp ucx::ucs)

  #endfunction()

  # call add_gpu_application here
  ## add_gpu_application("cutensorexp"  "exps" "cutensorexp")
  # add_gpu_application("test_alignment"  "exps" "test_alignment")

  #add_gpu_application("basicTest" "exps" "GPU_test")
  #add_gpu_application("exp1" "exps" "GPU_chain_mul")
  #add_gpu_application("exp2" "exps" "GPU_xtreme")
  #add_gpu_application("gpukm" "exps" "gpuexec_kernelmanager")
  add_gpu_llama_application("asd01" "asd01")
  add_gpu_llama_application("gpu_llama" "gpu")
  add_gpu_llama_application("gpu_llama_train" "gpu_train")
  add_gpu_llama_application("gpu_train_info" "gpu_training_info")
  add_gpu_llama_application("gpu_inference_info" "gpu_inference_info")
  add_gpu_llama_application("gpu_model_parallel" "gpu_model_parallel")
endif()

add_dependencies(einsummable proto)
