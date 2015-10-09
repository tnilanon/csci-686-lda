#include "error_code.h"
#include "dataset.h"

#define _[r][k] [(r)*K+(k)]

unsigned long * N_theta_d_k, * N_phi_w_k, * N_z_k;
unsigned long * N_hat_phi_w_k, * N_hat_z_k;

int main(int argc, char * argv[]) {

    if (argc != 4) {
        printf("Usage: ./fastLDA docword.txt iterations NumOfTopics\n");
        exit(INVALID_CALL);
    }

    num_iterations = strtoul(argv[2], NULL, 10);
    if (num_iterations == 0) {
        printf("Recheck number of iterations\n");
        exit(INVALID_NUM_ITERATIONS);
    }
    K = strtoul(argv[3], NULL, 10);
    if (K == 0) {
        printf("Recheck number of topics\n");
        exit(INVALID_NUM_TOPICS);
    }

    read_sparse_dataset(argv[1]);

    printf("first (%lu): %lu %lu\n", size_i[1], word_i_j[1][0], count_i_j[1][0]);
    printf("last (%lu): %lu %lu\n", size_i[D], word_i_j[D][size_i[D] - 1], count_i_j[D][size_i[D] - 1]);

    // Allocate calculation tables
    if ((N_theta_d_k = (unsigned long *) malloc((D + 1) * K * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_phi_w_k = (unsigned long *) malloc((W + 1) * K * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_z_k = (unsigned long *) malloc(K * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_hat_phi_w_k = (unsigned long *) malloc((W + 1) * K * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_hat_z_k = (unsigned long *) malloc(K * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }

    // Randomly initialize N_theta, N_phi, N_z

    // For each minibatch M
        // Set N_hat_phi, N_hat_z to zero
        // For each document i in M
            // For zero or more burn-in passes
                // For each token j
                    // Update gamma
                    // Update N_theta_i
            // For each token j
                // Update gamma
                // Update N_theta_i
                // Update N_hat_phi, N_hat_z
        // Update N_phi, N_z
}

