#include <ucp/api/ucp.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>

const size_t MSG_SIZE = 4096;
const int SERVER_PORT = 10086;
const size_t NUM_MESSAGES = 1000000; // Number of messages to receive

void receive_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info) {
    if (status != UCS_OK) {
        std::cerr << "Receive failed with status: " << ucs_status_string(status) << std::endl;
    }
    ucp_request_free(request);
}

int main() {
    setenv("UCX_TLS", "rc", 1); // Set RDMA transport
    setenv("UCX_NET_DEVICES", "mlx5_1:1", 1); // Set RDMA device

    ucp_params_t ucp_params;
    ucp_config_t *config;
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    ucp_worker_params_t worker_params;
    ucp_address_t *local_address;
    size_t local_address_length;

    // Initialize UCX context
    ucp_config_read(NULL, NULL, &config);
    std::memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG;

    ucp_init(&ucp_params, config, &ucp_context);
    ucp_config_release(config);

    // Create UCX worker
    std::memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    ucp_worker_create(ucp_context, &worker_params, &ucp_worker);

    // Get the local worker's address
    ucp_worker_get_address(ucp_worker, &local_address, &local_address_length);

    // Set up the server socket
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        std::cerr << "Failed to create server socket" << std::endl;
        return 1;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);

    if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind server socket" << std::endl;
        close(server_sock);
        return 1;
    }

    if (listen(server_sock, 1) < 0) {
        std::cerr << "Failed to listen on server socket" << std::endl;
        close(server_sock);
        return 1;
    }

    std::cout << "Server is listening on port " << SERVER_PORT << std::endl;

    // Accept a client connection
    int client_sock = accept(server_sock, NULL, NULL);
    if (client_sock < 0) {
        std::cerr << "Failed to accept client connection" << std::endl;
        close(server_sock);
        return 1;
    }

    // Send the local worker's address to the client
    if (send(client_sock, &local_address_length, sizeof(local_address_length), 0) < 0) {
        std::cerr << "Failed to send address length" << std::endl;
        close(client_sock);
        close(server_sock);
        return 1;
    }

    if (send(client_sock, local_address, local_address_length, 0) < 0) {
        std::cerr << "Failed to send local address" << std::endl;
        close(client_sock);
        close(server_sock);
        return 1;
    }

    close(client_sock);
    close(server_sock);

    // Allocate memory for receiving messages
    char *message = (char*)std::malloc(MSG_SIZE);
    std::memset(message, 0, MSG_SIZE);

    std::cout << "Waiting to receive " << NUM_MESSAGES << " messages..." << std::endl;

    // Receive messages from the client
    for (size_t i = 0; i < NUM_MESSAGES; i++) {
        void *recv_request;
        ucp_tag_recv_info_t info;
        recv_request = ucp_tag_recv_nb(ucp_worker, message, MSG_SIZE, ucp_dt_make_contig(1), 0, 0, receive_callback);

        if (UCS_PTR_IS_ERR(recv_request)) {
            std::cerr << "Receive request failed: " << ucs_status_string(UCS_PTR_STATUS(recv_request)) << std::endl;
            break;
        } else if (recv_request != NULL) {
            while (!ucp_request_is_completed(recv_request)) {
                ucp_worker_progress(ucp_worker);
            }
        }
    }

    std::cout << "Received " << NUM_MESSAGES << " messages" << std::endl;

    // Clean up
    std::free(message);
    ucp_worker_release_address(ucp_worker, local_address);
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);

    return 0;
}