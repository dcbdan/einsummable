#pragma once
#include "../base/setup.h"

#include <sys/socket.h>
#include <ucp/api/ucp.h>

void handle_ucs_error(ucs_status_t status, string msg = "");

struct _connection_t {
  _connection_t(int fd)
    : fd(fd)
  {}

  ~_connection_t();

  int const& get() const { return fd; }
private:
  int fd;
};

using connection_ptr_t = std::shared_ptr<_connection_t>;

connection_ptr_t connect_client(string const& server);

struct listen_server_t {
  listen_server_t();

  ~listen_server_t();

  connection_ptr_t connect();

private:
  int listen_fd;
};


