cmake_minimum_required(VERSION 3.2)

project(einsummable)

add_library(esummable STATIC
  src/copyregion.cc
  src/coster.cc
  src/einsummable.cc
  src/graph.cc
  src/indexer.cc
  src/partdim.cc
  src/partition.cc
  src/setup.cc
  src/taskgraph.cc
)
target_include_directories(esummable
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(make_3dmatmul exps/make_3dmatmul.cc)
target_link_libraries(make_3dmatmul esummable)

add_executable(reference exps/reference.cc src/reference.cc)
target_link_libraries(reference esummable)

add_executable(copyregion exps/copyregion.cc)
target_link_libraries(copyregion esummable)

add_executable(coster exps/coster.cc)
target_link_libraries(coster esummable)

