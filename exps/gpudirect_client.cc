#include <ucp/api/ucp.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <chrono>

const size_t MSG_SIZE = 4096;
const int SERVER_PORT = 10086;
const size_t NUM_MESSAGES = 1000000; // Number of messages to send

void send_callback(void *request, ucs_status_t status) {
    if (status != UCS_OK) {
        std::cerr << "Send failed with status: " << ucs_status_string(status) << std::endl;
    }
    ucp_request_free(request);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server_ip>" << std::endl;
        return 1;
    }

    setenv("UCX_TLS", "rc", 1); // Set RDMA transport
    setenv("UCX_NET_DEVICES", "mlx5_1:1", 1); // Set RDMA device

    ucp_params_t ucp_params;
    ucp_config_t *config;
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    ucp_worker_params_t worker_params;
    ucp_address_t *server_address;
    size_t server_address_length;

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

    // Connect to the server to get its UCX address
    int client_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);

    if (inet_pton(AF_INET, argv[1], &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid server IP address" << std::endl;
        close(client_sock);
        return 1;
    }

    if (connect(client_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect to server" << std::endl;
        close(client_sock);
        return 1;
    }

    // Receive the server's address
    if (recv(client_sock, &server_address_length, sizeof(server_address_length), 0) <= 0) {
        std::cerr << "Failed to receive address length" << std::endl;
        close(client_sock);
        return 1;
    }

    server_address = (ucp_address_t *)std::malloc(server_address_length);
    if (server_address == NULL) {
        std::cerr << "Failed to allocate memory for server address" << std::endl;
        close(client_sock);
        return 1;
    }

    if (recv(client_sock, server_address, server_address_length, 0) <= 0) {
        std::cerr << "Failed to receive server address" << std::endl;
        close(client_sock);
        std::free(server_address);
        return 1;
    }

    close(client_sock);

    // Create the UCX endpoint
    ucp_ep_h ep;
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = server_address;
    ucp_ep_create(ucp_worker, &ep_params, &ep);

    // Send the messages and measure performance
    char *message = (char*)std::malloc(MSG_SIZE);
    std::memset(message, 0, MSG_SIZE);

    std::cout << "Sending " << NUM_MESSAGES << " messages..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_MESSAGES; i++) {
        void *send_request;
        send_request = ucp_tag_send_nb(ep, message, MSG_SIZE, ucp_dt_make_contig(1), 0, send_callback);

        if (UCS_PTR_IS_ERR(send_request)) {
            std::cerr << "Send request failed: " << ucs_status_string(UCS_PTR_STATUS(send_request)) << std::endl;
            break;
        } else if (send_request != NULL) {
            while (!ucp_request_is_completed(send_request)) {
                ucp_worker_progress(ucp_worker);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    double total_data = static_cast<double>(MSG_SIZE * NUM_MESSAGES) / (1024 * 1024); // Data in MB
    double bandwidth = total_data / duration.count(); // Bandwidth in MB/s

    std::cout << "Sent " << NUM_MESSAGES << " messages in " << duration.count() << " seconds." << std::endl;
    std::cout << "Achieved bandwidth: " << bandwidth << " MB/s" << std::endl;

    // Clean up
    std::free(message);
    std::free(server_address);
    ucp_ep_destroy(ep);
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);

    return 0;
}
