#include "comm_util.h"

#include <sys/poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>

#include <arpa/inet.h>

#include <thread>

#define CHKERR_ACTION(_cond, _msg, _action)                                                        \
    do {                                                                                           \
        if (_cond) {                                                                               \
            fprintf(stderr, "Failed to %s\n", _msg);                                               \
            _action;                                                                               \
        }                                                                                          \
    } while (0)

#define CHKERR_JUMP(_cond, _msg, _label) CHKERR_ACTION(_cond, _msg, goto _label)

void handle_ucs_error(ucs_status_t status, string msg)
{
    if (status != UCS_OK) {
        if (msg == "") {
            msg = "ucp error";
        }
        throw std::runtime_error(msg + ": " + write_with_ss(status));
    }
}

static uint16_t    server_port = 13337;
static sa_family_t ai_family = AF_INET;

_connection_t::~_connection_t()
{
    close(fd);
}

listen_server_t::~listen_server_t()
{
    close(listen_fd);
}

int _connect_client(string server_)
{
    auto server = server_.c_str();

    int             sockfd = -1;
    char            service[8];
    struct addrinfo hints, *res, *t;
    int             ret;

    snprintf(service, sizeof(service), "%u", server_port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags = (server == NULL) ? AI_PASSIVE : 0;
    hints.ai_family = ai_family;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(server, service, &hints, &res);
    CHKERR_JUMP(ret < 0, "getaddrinfo() failed", out);

    for (t = res; t != NULL; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd < 0) {
            continue;
        }

        if (connect(sockfd, t->ai_addr, t->ai_addrlen) == 0) {
            break;
        }

        close(sockfd);
        sockfd = -1;
    }

    CHKERR_ACTION(sockfd < 0, "open client socket", (void)sockfd /* no action */);

out_free_res:
    freeaddrinfo(res);
out:
    return sockfd;
err_close_sockfd:
    close(sockfd);
    sockfd = -1;
    goto out_free_res;
}

connection_ptr_t connect_client(string const& server)
{
    int oob_sock;
    for (int i = 0; i != 30; ++i) {
        using namespace std::chrono_literals;
        oob_sock = _connect_client(server);
        if (oob_sock >= 0) {
            break;
        }
        std::this_thread::sleep_for(2000ms);
    }
    if (oob_sock < 0) {
        throw std::runtime_error("could not connect");
    }
    return std::make_shared<_connection_t>(oob_sock);
}

int _connect_listen()
{
    int             listenfd = -1;
    int             optval = 1;
    char            service[8];
    struct addrinfo hints, *res, *t;
    int             ret;

    snprintf(service, sizeof(service), "%u", server_port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags = AI_PASSIVE;
    hints.ai_family = ai_family;
    hints.ai_socktype = SOCK_STREAM;

    ret = getaddrinfo(NULL, service, &hints, &res);
    CHKERR_JUMP(ret < 0, "getaddrinfo() failed", out);

    for (t = res; t != NULL; t = t->ai_next) {
        listenfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (listenfd < 0) {
            continue;
        }

        ret = setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
        CHKERR_JUMP(ret < 0, "server setsockopt()", err_close_sockfd);

        if (bind(listenfd, t->ai_addr, t->ai_addrlen) == 0) {
            ret = listen(listenfd, 0);
            CHKERR_JUMP(ret < 0, "listen server", err_close_sockfd);
            break;
        }

        close(listenfd);
        listenfd = -1;
    }

    CHKERR_ACTION(listenfd < 0, "open server socket", (void)listenfd /* no action */);

out_free_res:
    freeaddrinfo(res);
out:
    return listenfd;
err_close_sockfd:
    close(listenfd);
    listenfd = -1;
    goto out_free_res;
}

listen_server_t::listen_server_t()
{
    listen_fd = _connect_listen();
    if (listen_fd < 0) {
        throw std::runtime_error("listen should not be negative");
    }
}

connection_ptr_t listen_server_t::connect()
{
    int ret = accept(listen_fd, NULL, NULL);
    if (ret < 0) {
        throw std::runtime_error("connect listen: got negative..");
    }
    return std::make_shared<_connection_t>(ret);
}
