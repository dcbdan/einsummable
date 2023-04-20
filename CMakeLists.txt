cmake_minimum_required(VERSION 3.2)

project(einsummable)

add_library(einsummable STATIC
  src/einsummable/copyregion.cc
  src/einsummable/coster.cc
  src/einsummable/einsummable.cc
  src/einsummable/graph.cc
  src/einsummable/indexer.cc
  src/einsummable/partdim.cc
  src/einsummable/partition.cc
  src/einsummable/reference.cc
  src/einsummable/setup.cc
  src/einsummable/taskgraph.cc
)
target_include_directories(einsummable
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src/einsummable
)

add_executable(make_3dmatmul exps/make_3dmatmul.cc)
target_link_libraries(make_3dmatmul einsummable)

add_executable(reference exps/reference.cc)
target_link_libraries(reference einsummable)

add_executable(copyregion exps/copyregion.cc)
target_link_libraries(copyregion einsummable)

add_executable(coster exps/coster.cc)
target_link_libraries(coster einsummable)

