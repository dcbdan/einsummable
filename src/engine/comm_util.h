#pragma once
#include "../base/setup.h"

#include <sys/socket.h>
#include <ucp/api/ucp.h>

int connect_common(const char *server, uint16_t server_port, sa_family_t af);

void handle_ucs_error(ucs_status_t status, string msg = "");


