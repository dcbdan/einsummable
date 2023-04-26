#pragma once
#include "../../einsummable/setup.h"
#include "../../einsummable/reference.h" // buffer

#include <mpi.h>

struct mpi_t {
  mpi_t(int argc, char** argv);

  ~mpi_t();

  void barrier();

  void recv(buffer_t buffer, int src, int tag);
  void send(buffer_t buffer, int dst, int tag);

  void send_str(string const& x, int dst);
  string recv_str(int src);

  void send_int(int val, int dst, int tag);
  int recv_int_from_anywhere(int tag);

  int this_rank;
  int world_size;
  int max_tag;

private:
   void _send_recv(
    bool is_send,
    buffer_t buffer,
    int loc,
    int tag);

public:
  template <typename T>
  void send_vector(vector<T> const& xs, int dst) {
    int n = xs.size();
    MPI_Send(&n, 1, MPI_INT,
      dst, 0, MPI_COMM_WORLD);
    MPI_Send((void*)xs.data(), n*sizeof(T), MPI_CHAR,
      dst, 0, MPI_COMM_WORLD);
  }

  template <typename T>
  vector<T> recv_vector(int src) {
    int n;
    MPI_Recv(&n, 1, MPI_INT, src,
      0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<T> ret(n);
    MPI_Recv((void*)ret.data(), n*sizeof(T), MPI_CHAR, src,
      0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return ret;
  }
};


