#include "comm_util.h"

#include <sys/poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>

#define CHKERR_ACTION(_cond, _msg, _action) \
    do { \
        if (_cond) { \
            fprintf(stderr, "Failed to %s\n", _msg); \
            _action; \
        } \
    } while (0)


#define CHKERR_JUMP(_cond, _msg, _label) \
    CHKERR_ACTION(_cond, _msg, goto _label)

// Stolen from ucx hello world tutorial
int connect_common(const char *server, uint16_t server_port, sa_family_t af)
{
    int sockfd   = -1;
    int listenfd = -1;
    int optval   = 1;
    char service[8];
    struct addrinfo hints, *res, *t;
    int ret;

    snprintf(service, sizeof(service), "%u", server_port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags    = (server == NULL) ? AI_PASSIVE : 0;
    hints.ai_family   = af;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(server, service, &hints, &res);
    CHKERR_JUMP(ret < 0, "getaddrinfo() failed", out);

    for (t = res; t != NULL; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd < 0) {
            continue;
        }

        if (server != NULL) {
            if (connect(sockfd, t->ai_addr, t->ai_addrlen) == 0) {
                break;
            }
        } else {
            ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,
                             sizeof(optval));
            CHKERR_JUMP(ret < 0, "server setsockopt()", err_close_sockfd);

            if (bind(sockfd, t->ai_addr, t->ai_addrlen) == 0) {
                ret = listen(sockfd, 0);
                CHKERR_JUMP(ret < 0, "listen server", err_close_sockfd);

                /* Accept next connection */
                fprintf(stdout, "Waiting for connection...\n");
                listenfd = sockfd;
                sockfd   = accept(listenfd, NULL, NULL);
                close(listenfd);
                break;
            }
        }

        close(sockfd);
        sockfd = -1;
    }

    CHKERR_ACTION(sockfd < 0,
                  (server) ? "open client socket" : "open server socket",
                  (void)sockfd /* no action */);

out_free_res:
    freeaddrinfo(res);
out:
    return sockfd;
err_close_sockfd:
    close(sockfd);
    sockfd = -1;
    goto out_free_res;
}

void handle_ucs_error(ucs_status_t status, string msg) {
  if(status != UCS_OK) {
    if(msg == "") {
      msg = "ucp error";
    }
    throw std::runtime_error(msg + ": " + write_with_ss(status));
  }
}

