#include "error_code.h"
#include "dataset.h"
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#define N_theta_d_k(d, k) N_theta_d_k[(d)*K+(k)]
#define N_phi_w_k(w, k) N_phi_w_k[(w)*K+(k)]
#define N_z_k(k) N_z_k[(k)]
#define N_hat_phi_t_w_k(t, w, k) N_hat_phi_t_w_k[(t)][(w)*K+(k)]
#define N_hat_z_t_k(t, k) N_hat_z_t_k[(t)][(k)]

#define N_count_d(d) N_count_d[(d)]
#define theta_d_k(d, k) theta_d_k[(d)*K+(k)]
#define phi_w_k(w, k) phi_w_k[(w)*K+(k)]

const double alpha = 0.5;
const double eta = 0.5;

long number_of_threads = 12;
long batch_size = 500;

double * N_theta_d_k, * N_phi_w_k, * N_z_k;
long * C_t;
double ** N_hat_phi_t_w_k, ** N_hat_z_t_k;
double * N_count_d, * theta_d_k, * phi_w_k;

struct timeval tic, toc, diff;

// http://www.gnu.org/software/libc/manual/html_node/Elapsed-Time.html
/* Subtract the ‘struct timeval’ values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0. */
int timeval_subtract (result, x, y)
    struct timeval *result, *x, *y;
{
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }
    /* Compute the time remaining to wait. tv_usec is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;
    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}

void calculate_theta_phi();
void calculate_perplexity();
void inference(long iteration_idx);

int main(int argc, char * argv[]) {

    if (argc != 4) {
        printf("Usage: ./fastLDA docword.txt iterations NumOfTopics\n");
        exit(INVALID_CALL);
    }

    num_iterations = strtol(argv[2], NULL, 10);
    if (num_iterations == 0) {
        printf("Recheck number of iterations\n");
        exit(INVALID_NUM_ITERATIONS);
    }
    K = strtol(argv[3], NULL, 10);
    if (K == 0) {
        printf("Recheck number of topics\n");
        exit(INVALID_NUM_TOPICS);
    }

    gettimeofday(&tic, NULL);
    read_sparse_dataset(argv[1]);
    gettimeofday(&toc, NULL);
    timeval_subtract(&diff, &toc, &tic);
    printf("reading file took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
    printf("\n");

    printf("first word (%ld distinct in doc): %ld %ld\n", size_d[1], word_d_i[1][0], count_d_i[1][0]);
    printf("last word (%ld distinct in doc): %ld %ld\n", size_d[D], word_d_i[D][size_d[D] - 1], count_d_i[D][size_d[D] - 1]);
    printf("\n");

    // allocate calculation tables
    gettimeofday(&tic, NULL);

    if ((N_theta_d_k = (double *) malloc((D + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_phi_w_k = (double *) malloc((W + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    N_z_k = N_phi_w_k;

    C_t = (long *) malloc(number_of_threads * sizeof(long));

    if ((N_hat_phi_t_w_k = (double **) malloc(number_of_threads * sizeof(double *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    for (long t = 0; t < number_of_threads; ++t) {
        if ((N_hat_phi_t_w_k[t] = (double *) malloc((W + 1) * K * sizeof(double))) == NULL) {
            printf("Out of memory\n");
            exit(OUT_OF_MEMORY);
        }
    }
    N_hat_z_t_k = N_hat_phi_t_w_k;

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

    gettimeofday(&toc, NULL);
    timeval_subtract(&diff, &toc, &tic);
    printf("memory allocation took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);

    // randomly initialize N_theta_d_k, N_phi_w_k, N_z_k
    gettimeofday(&tic, NULL);
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
    gettimeofday(&toc, NULL);
    timeval_subtract(&diff, &toc, &tic);
    printf("random initialization took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
    double sum = 0;
    for (long k = 0; k < K; ++k) {
        sum += N_z_k(k);
    }
    printf("sum(N_z_k): %.0f\n", sum);
    printf("\n");

    // calculate average perplexity per word
    printf("calculate initial perplexity:\n");
    gettimeofday(&tic, NULL);
    calculate_theta_phi();
    gettimeofday(&toc, NULL);
    timeval_subtract(&diff, &toc, &tic);
    printf("theta and phi calculation took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
    gettimeofday(&tic, NULL);
    calculate_perplexity();
    gettimeofday(&toc, NULL);
    timeval_subtract(&diff, &toc, &tic);
    printf("perplexity calculation took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
    printf("\n");

    // for each iteration
    for (long iteration_idx = 1; iteration_idx <= num_iterations; ++iteration_idx) {
        printf("iteration %ld:\n", iteration_idx);
        gettimeofday(&tic, NULL);
        inference(iteration_idx);
        gettimeofday(&toc, NULL);
        timeval_subtract(&diff, &toc, &tic);
        printf("inference took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
        gettimeofday(&tic, NULL);
        calculate_theta_phi();
        gettimeofday(&toc, NULL);
        timeval_subtract(&diff, &toc, &tic);
        printf("theta and phi calculation took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
        gettimeofday(&tic, NULL);
        calculate_perplexity();
        gettimeofday(&toc, NULL);
        timeval_subtract(&diff, &toc, &tic);
        printf("perplexity calculation took %.3f seconds\n", diff.tv_sec + (double) diff.tv_usec / 1000000);
        printf("\n");
    }

    return 0;
}

void calculate_theta_phi() {
    #pragma omp parallel for schedule(static) num_threads(number_of_threads)
    for (long d = 1; d <= D; ++d) {
        N_count_d(d) = 0;
        for (long k = 0; k < K; ++k) {
            N_count_d(d) += N_theta_d_k(d, k);
        }
    }
    #pragma omp parallel for schedule(static) num_threads(number_of_threads)
    for (long d = 1; d <= D; ++d) {
        for (long k = 0; k < K; ++k) {
            theta_d_k(d, k) = (double)(N_theta_d_k(d, k) + alpha) / (N_count_d(d) + K * alpha);
        }
    }
    #pragma omp parallel for schedule(static) num_threads(number_of_threads)
    for (long w = 1; w <= W; ++w) {
        for (long k = 0; k < K; ++k) {
            phi_w_k(w, k) = (double)(N_phi_w_k(w, k) + eta) / (N_z_k(k) + W * eta);
        }
    }
}

void calculate_perplexity() {
    double entropy = 0;
    #pragma omp parallel for reduction(+:entropy) schedule(static) num_threads(number_of_threads)
    for (long d = 1; d <= D; ++d) {
        for (long i = 0; i < size_d[d]; ++i) {
            double P_d_i = 0;
            for (long k = 0; k < K; ++k) {
                P_d_i += theta_d_k(d, k) * phi_w_k(word_d_i[d][i], k);
            }
            entropy += count_d_i[d][i] * log2(P_d_i);
        }
    }
    entropy = - entropy;
    printf("entropy (per word), perplexity: %.2f (%.2f), %.2f\n", entropy, entropy / C, exp2(entropy / C));
}

void inference(long iteration_idx) {
    double rho_theta = 1.0 / pow(100 + 10 * iteration_idx, 0.9);
    double rho_phi = 1.0 / pow(100 + 10 * iteration_idx, 0.9);

    long num_batches = ceil((double)D / batch_size);
    long num_epochs = ceil((double)num_batches / number_of_threads);

    for (long epoch_id = 0; epoch_id < num_epochs; ++epoch_id) {
        long first_batch_this_epoch = epoch_id * number_of_threads;
        long first_batch_next_epoch = (epoch_id + 1) * number_of_threads;
        if (first_batch_next_epoch > num_batches) {
            first_batch_next_epoch = num_batches;
        }
        long num_batches_this_epoch = first_batch_next_epoch - first_batch_this_epoch;

        // for each batch in epoch
        #pragma omp parallel num_threads(num_batches_this_epoch)
        {
            long thread_id = omp_get_thread_num();
            long batch_id = thread_id + epoch_id * number_of_threads;
            long first_doc_this_batch = batch_id * batch_size + 1;
            long first_doc_next_batch = (batch_id + 1) * batch_size + 1;
            if (first_doc_next_batch > D + 1) {
                first_doc_next_batch = D + 1;
            }

            C_t[thread_id] = 0;
            // set N_hat_phi_t_w_k, N_hat_z_t_k to zero
            memset(N_hat_phi_t_w_k[thread_id], 0, (W + 1) * K * sizeof(double));

            double * gamma_k = (double *)malloc(K * sizeof(double));

            // for each document d in batch
            for (long d = first_doc_this_batch; d < first_doc_next_batch; ++d) {
                C_t[thread_id] += C_d[d];

                // for zero or more burn-in passes
                // for each token i
                for (long i = 0; i < size_d[d]; ++i) {
                    // update gamma
                    double sum = 0;
                    for (long k = 0; k < K; ++k) {
                        gamma_k[k] = (N_theta_d_k(d, k) + alpha) \
                            * (N_phi_w_k(word_d_i[d][i], k) + eta) \
                            / (N_z_k(k) + W * eta);
                        sum += gamma_k[k];
                    }
                    for (long k = 0; k < K; ++k) {
                        gamma_k[k] /= sum;
                    }
                    // update N_theta_d_k
                    double factor = pow(1 - rho_theta, count_d_i[d][i]);
                    for (long k = 0; k < K; ++k) {
                        N_theta_d_k(d, k) = factor * N_theta_d_k(d, k) \
                            + (1 - factor) * C_d[d] * gamma_k[k];
                    }
                }

                // for each token i
                for (long i = 0; i < size_d[d]; ++i) {
                    // update gamma
                    double sum = 0;
                    for (long k = 0; k < K; ++k) {
                        gamma_k[k] = (N_theta_d_k(d, k) + alpha) \
                            * (N_phi_w_k(word_d_i[d][i], k) + eta) \
                            / (N_z_k(k) + W * eta);
                        sum += gamma_k[k];
                    }
                    for (long k = 0; k < K; ++k) {
                        gamma_k[k] /= sum;
                    }
                    // update N_theta_d_k
                    double factor = pow(1 - rho_theta, count_d_i[d][i]);
                    for (long k = 0; k < K; ++k) {
                        N_theta_d_k(d, k) = factor * N_theta_d_k(d, k) \
                            + (1 - factor) * C_d[d] * gamma_k[k];
                    }
                    // update N_hat_phi_t_w_k, N_hat_z_t_k
                    for (long k = 0; k < K; ++k) {
                        double temp = count_d_i[d][i] * gamma_k[k];
                        N_hat_phi_t_w_k(thread_id, word_d_i[d][i], k) += temp;
                        N_hat_z_t_k(thread_id, k) += temp;
                    }
                }
            }

            free(gamma_k);
        } // end omp parallel
        // update N_phi_w_k, N_z_k
        #pragma omp parallel for collapse(2) schedule(static) num_threads(number_of_threads)
        for (long w = 1; w <= W; ++w) {
            for (long k = 0; k < K; ++k) {
                for (long t = 0; t < num_batches_this_epoch; ++t) {
                    N_phi_w_k(w, k) = rho_phi * C / C_t[t] * N_hat_phi_t_w_k(t, w, k) \
                        + (1 - rho_phi) * N_phi_w_k(w, k);
                }
            }
        }
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (long k = 0; k < K; ++k) {
            for (long t = 0; t < num_batches_this_epoch; ++t) {
                N_z_k(k) = rho_phi * C / C_t[t] * N_hat_z_t_k(t, k) \
                    + (1 - rho_phi) * N_z_k(k);
            }
        }
    }
}

