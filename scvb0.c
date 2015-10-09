#include "error_code.h"
#include "dataset.h"

#define N_theta_d_k(d, k) N_theta_d_k[(d)*K+(k)]
#define N_phi_w_k(w, k) N_phi_w_k[(w)*K+(k)]
#define N_z_k(k) N_z_k[(k)]
#define N_hat_phi_w_k(w, k) N_hat_phi_w_k[(w)*K+(k)]
#define N_hat_z_k(k) N_hat_z_k[(k)]

#define N_count_d(d) N_count_d[(d)]
#define theta_d_k(d, k) theta_d_k[(d)*K+(k)]
#define phi_w_k(w, k) phi_w_k[(w)*K+(k)]

const double alpha = 0.5;
const double eta = 0.5;

double * N_theta_d_k, * N_phi_w_k, * N_z_k;
double * N_hat_phi_w_k, * N_hat_z_k;
double * N_count_d, * theta_d_k, * phi_w_k;

clock_t tic, toc;

void calculate_theta_phi();

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

    tic = clock();
    read_sparse_dataset(argv[1]);
    toc = clock();
    printf("reading file took %.3f seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);
    printf("\n");

    printf("first (%lu): %lu %lu\n", size_d[1], word_d_i[1][0], count_d_i[1][0]);
    printf("last (%lu): %lu %lu\n", size_d[D], word_d_i[D][size_d[D] - 1], count_d_i[D][size_d[D] - 1]);

    // Allocate calculation tables
    tic = clock();

    if ((N_theta_d_k = (double *) malloc((D + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_phi_w_k = (double *) malloc((W + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    N_z_k = N_phi_w_k;
    if ((N_hat_phi_w_k = (double *) malloc((W + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    N_hat_z_k = N_hat_phi_w_k;

    if ((N_count_d = (double *) malloc((D + 1) * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((theta_d_k = (double *) malloc((D + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((phi_w_k = (double *) malloc((W + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }

    toc = clock();
    printf("memory allocation took %.3f seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);

    // Randomly initialize N_theta_d_k, N_phi_w_k, N_z_k
    tic = clock();
    memset(N_theta_d_k, 0, (D + 1) * K * sizeof(double));
    memset(N_phi_w_k, 0, (W + 1) * K * sizeof(double));
    // N_z_k is in N_phi_w_k
    for (long d = 1; d <= D; ++d) {
        for (long i = 0; i < size_d[d]; ++i) {
            for (long c = 0; c < count_d_i[d][i]; ++c) {
                long k = rand() % K;
                N_theta_d_k(d, k) += 1;
                N_phi_w_k(word_d_i[d][i], k) += 1;
                N_z_k(k) += 1;
            }
        }
    }
    toc = clock();
    printf("random initialization took %.3f seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);
    double sum = 0;
    for (long k = 0; k < K; ++k) {
        sum += N_z_k(k);
    }
    printf("sum(N_z_k): %.0f\n", sum);
    printf("\n");

    // Calculate average perplexity per word
    tic = clock();
    calculate_theta_phi();
    toc = clock();
    printf("theta and phi calculation took %.3f seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);
    tic = clock();
    double entropy = 0;
    for (long d = 1; d <= D; ++d) {
        for (long i = 0; i < size_d[d]; ++i) {
            double sum = 0;
            for (long k = 0; k < K; ++k) {
                sum += theta_d_k(d, k) * phi_w_k(word_d_i[d][i], k);
            }
            entropy += count_d_i[d][i] * log2(sum);
        }
    }
    entropy = - entropy;
    printf("entropy (per word), perplexity: %.2f (%.2f), %.2f\n", entropy, entropy / N, exp2(entropy / N));
    toc = clock();
    printf("perplexity calculation took %.3f seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);

    // For each minibatch M
        // Set N_hat_phi_w_k, N_hat_z_k to zero
        // For each document i in M
            // For zero or more burn-in passes
                // For each token j
                    // Update gamma
                    // Update N_theta_d_k
            // For each token j
                // Update gamma
                // Update N_theta_d_k
                // Update N_hat_phi_w_k, N_hat_z_k
        // Update N_phi_w_k, N_z_k
}

void calculate_theta_phi() {
    for (long d = 1; d <= D; ++d) {
        N_count_d(d) = 0;
        for (long k = 0; k < K; ++k) {
            N_count_d(d) += N_theta_d_k(d, k);
        }
    }
    for (long d = 1; d <= D; ++d) {
        for (long k = 0; k < K; ++k) {
            theta_d_k(d, k) = (double)(N_theta_d_k(d, k) + alpha) / (N_count_d(d) + K * alpha);
        }
    }
    for (long w = 1; w <= W; ++w) {
        for (long k = 0; k < K; ++k) {
            phi_w_k(w, k) = (double)(N_phi_w_k(w, k) + eta) / (N_z_k(k) + W * eta);
        }
    }
}

