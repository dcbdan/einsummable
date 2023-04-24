#include "mpi_class.h"

mpi_t::mpi_t(int argc, char** argv)
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if(provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_Init_thread");
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Note: We could set MPI to throw errors C++ style
  //   MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI::ERRORS_THROW_EXCEPTIONS);
  // But it looks like with MPI_Issend and MPI_Irecv, the error
  // doesn't get caught. (The test case: when doing an MPI_Issend to loc 1 when
  // world size is only 1, the error wasn't caught)
}

mpi_t::~mpi_t()
{
  MPI_Finalize();
}

void mpi_t::barrier()
{
  MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_t::recv(buffer_t buffer, int src, int tag)
{
  return _send_recv(false, buffer, src, tag);
}

void mpi_t::send(buffer_t buffer, int dst, int tag)
{
  return _send_recv(true, buffer, dst, tag);
}

void mpi_t::_send_recv(
  bool is_send,
  buffer_t buffer,
  int loc,
  int tag)
{
  int64_t const max_size = 1073741824 / sizeof(float); // 2^30 / itemsize
  uint64_t numel_rem = buffer->size;
  void* data = (void*)(buffer->data);
  while(numel_rem > 0) {
    int num_to_comm = numel_rem < max_size ? numel_rem : max_size;

    int result = is_send
        ? MPI_Ssend(data, num_to_comm, MPI_FLOAT, loc, tag, MPI_COMM_WORLD)
        : MPI_Recv( data, num_to_comm, MPI_FLOAT, loc, tag, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE)
        ;
    if(result != MPI_SUCCESS) {
      throw std::runtime_error("Invalid send or recv");
    }

    data = (void*)((char*)data + (sizeof(float) * num_to_comm));
    numel_rem -= num_to_comm;
  }
}

void mpi_t::send_str(std::string const& x, int dst)
{
  int n = x.size();
  MPI_Send(&n, 1, MPI_INT,
    dst, -1, MPI_COMM_WORLD);
  MPI_Send((void*)x.data(), n, MPI_CHAR,
    dst, -1, MPI_COMM_WORLD);
}

std::string mpi_t::recv_str(int src)
{
  int n;
  MPI_Recv(&n, 1, MPI_INT, src,
    -1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  vector<char> ret(n);
  MPI_Recv((void*)ret.data(), n, MPI_CHAR, src,
    -1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return std::string(ret.begin(), ret.end());
}

