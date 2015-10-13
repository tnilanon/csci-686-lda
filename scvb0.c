#include "error_code.h"
#include "dataset.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "omp.h"

// inference
#define N_theta_d_k(d, k) N_theta_d_k[(d)*K+(k)]
#define N_phi_w_k(w, k) N_phi_w_k[(w)*K+(k)]
#define N_z_k(k) N_z_k[(k)]
#define N_hat_phi_t_w_k(t, w, k) N_hat_phi_t_w_k[(t)][(w)*K+(k)]
#define N_hat_z_t_k(t, k) N_hat_z_t_k[(t)][(k)]

// recover hidden variables
#define N_count_d(d) N_count_d[(d)]
#define theta_d_k(d, k) theta_d_k[(d)*K+(k)]
#define phi_w_k(w, k) phi_w_k[(w)*K+(k)]

#ifndef NDEBUG
#ifndef REPORT_PERPLEXITY
#define REPORT_PERPLEXITY
#endif
#endif

// constants
#ifndef ALPHA
#define ALPHA 0.5
#endif
#ifndef ETA
#define ETA 0.5
#endif
#define BATCH_SIZE 500
#define MIN_NUM_THREADS 12
#define NUM_TERMS_REPORTED_PER_TOPIC 100

// will be set by omp
long _num_threads_;

// input
long num_iterations;
long K;

// for calculations
double * N_theta_d_k, * N_phi_w_k, * N_z_k;
long * C_t;
double ** N_hat_phi_t_w_k, ** N_hat_z_t_k;
double * N_count_d, * theta_d_k, * phi_w_k;

// for output
struct _word_probability ** topic;

void calculate_theta_phi();
void calculate_perplexity();
void inference(long iteration_idx);
void calculate_topic();
void output();

int main(int argc, char * argv[]) {

    if (argc < 4 || argc > 5) {
        printf("Usage: ./fastLDA docword.txt iterations num_topics [num_threads]\n");
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

    if (argc == 5) {
        _num_threads_ = strtol(argv[4], NULL, 10);
        if (_num_threads_ == 0) {
            printf("Recheck number of threads\n");
            exit(INVALID_NUM_THREADS);
        }
    } else {
        _num_threads_ = 0;
    }

    if (_num_threads_ == 0) {
        long num_processors = omp_get_num_procs();
        _num_threads_ = floor(0.9 * num_processors);
        if (_num_threads_ < MIN_NUM_THREADS) {
            _num_threads_ = MIN_NUM_THREADS;
        }
        printf("found %ld processors; set number of threads to %ld;\n", num_processors, _num_threads_);
    } else {
        printf("set number of threads to %ld;\n", _num_threads_);
    }
    printf("\n");

    start_timer();
    read_sparse_dataset(argv[1]);
    stop_timer("reading file took %.3f seconds\n");
    printf("\n");

#ifndef NDEBUG
    printf("first word (%ld distinct in doc %ld): id %ld count %ld\n", size_d[1], (long)1, word_d_i[1][0], count_d_i[1][0]);
    printf("last word (%ld distinct in doc %ld): id %ld count %ld\n", size_d[D], D, word_d_i[D][size_d[D] - 1], count_d_i[D][size_d[D] - 1]);
    printf("\n");
#endif

    // allocate calculation tables
    start_timer();

    if ((N_theta_d_k = (double *) malloc((D + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((N_phi_w_k = (double *) malloc((W + 1) * K * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    N_z_k = N_phi_w_k;

    C_t = (long *) malloc(_num_threads_ * sizeof(long));

    if ((N_hat_phi_t_w_k = (double **) malloc(_num_threads_ * sizeof(double *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    for (long t = 0; t < _num_threads_; ++t) {
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

    stop_timer("memory allocation took %.3f seconds\n");

    // randomly initialize N_theta_d_k, N_phi_w_k, N_z_k
    start_timer();
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
    stop_timer("random initialization took %.3f seconds\n");

#ifndef NDEBUG
    double sum = 0;
    for (long k = 0; k < K; ++k) {
        sum += N_z_k(k);
    }
    printf("sum(N_z_k): %.0f (should be equal to C)\n", sum);
#endif

    printf("\n");

#ifdef REPORT_PERPLEXITY
    // calculate average perplexity per word
    printf("calculate initial perplexity:\n");

    start_timer();
    calculate_theta_phi();
    stop_timer("theta and phi calculation took %.3f seconds\n");

    start_timer();
    calculate_perplexity();
    stop_timer("perplexity calculation took %.3f seconds\n");
    printf("\n");
#endif

    // for each iteration
    for (long iteration_idx = 1; iteration_idx <= num_iterations; ++iteration_idx) {
        printf("iteration %ld:\n", iteration_idx);

        start_timer();
        inference(iteration_idx);
        stop_timer("inference took %.3f seconds\n");

#ifdef REPORT_PERPLEXITY
        start_timer();
        calculate_theta_phi();
        stop_timer("theta and phi calculation took %.3f seconds\n");

        start_timer();
        calculate_perplexity();
        stop_timer("perplexity calculation took %.3f seconds\n");
#endif

        printf("\n");
    }

    start_timer();
    calculate_theta_phi();
    stop_timer("theta and phi calculation took %.3f seconds\n");

    start_timer();
    calculate_topic();
    stop_timer("topic calculation took %.3f seconds\n");

    start_timer();
    output();
    stop_timer("writing files took %.3f seconds\n");

    return 0;
}

void calculate_theta_phi() {
    #pragma omp parallel for schedule(static) num_threads(_num_threads_)
    for (long d = 1; d <= D; ++d) {
        N_count_d(d) = 0;
        for (long k = 0; k < K; ++k) {
            N_count_d(d) += N_theta_d_k(d, k);
        }
    }
    #pragma omp parallel for schedule(static) num_threads(_num_threads_)
    for (long d = 1; d <= D; ++d) {
        for (long k = 0; k < K; ++k) {
            theta_d_k(d, k) = (double)(N_theta_d_k(d, k) + ALPHA) / (N_count_d(d) + K * ALPHA);
        }
    }
    #pragma omp parallel for schedule(static) num_threads(_num_threads_)
    for (long w = 1; w <= W; ++w) {
        for (long k = 0; k < K; ++k) {
            phi_w_k(w, k) = (double)(N_phi_w_k(w, k) + ETA) / (N_z_k(k) + W * ETA);
        }
    }
}

void calculate_perplexity() {
    double entropy = 0;
    #pragma omp parallel for reduction(+:entropy) schedule(static) num_threads(_num_threads_)
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
    printf("(per word) entropy, perplexity: %.2f, %.2f\n", entropy / C, exp2(entropy / C));
}

void inference(long iteration_idx) {
    double rho_theta = pow(100 + 10 * iteration_idx, -0.9);
    double rho_phi = pow(100 + 10 * iteration_idx, -0.9);

    long num_batches = ceil((double)D / BATCH_SIZE);
    long num_epochs = ceil((double)num_batches / _num_threads_);

    for (long epoch_id = 0; epoch_id < num_epochs; ++epoch_id) {
        long first_batch_this_epoch = epoch_id * _num_threads_;
        long first_batch_next_epoch = (epoch_id + 1) * _num_threads_;
        if (first_batch_next_epoch > num_batches) {
            first_batch_next_epoch = num_batches;
        }
        long num_batches_this_epoch = first_batch_next_epoch - first_batch_this_epoch;

        // for each batch in epoch
        #pragma omp parallel num_threads(num_batches_this_epoch)
        {
            long thread_id = omp_get_thread_num();
            long batch_id = thread_id + epoch_id * _num_threads_;
            long first_doc_this_batch = batch_id * BATCH_SIZE + 1;
            long first_doc_next_batch = (batch_id + 1) * BATCH_SIZE + 1;
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
                        gamma_k[k] = (N_theta_d_k(d, k) + ALPHA) \
                            * (N_phi_w_k(word_d_i[d][i], k) + ETA) \
                            / (N_z_k(k) + W * ETA);
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
                        gamma_k[k] = (N_theta_d_k(d, k) + ALPHA) \
                            * (N_phi_w_k(word_d_i[d][i], k) + ETA) \
                            / (N_z_k(k) + W * ETA);
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

        // update N_phi_w_k
        #pragma omp parallel for collapse(2) schedule(static) num_threads(_num_threads_)
        for (long w = 1; w <= W; ++w) {
            for (long k = 0; k < K; ++k) {
                for (long t = 0; t < num_batches_this_epoch; ++t) {
                    N_phi_w_k(w, k) = rho_phi * C / C_t[t] * N_hat_phi_t_w_k(t, w, k) \
                        + (1 - rho_phi) * N_phi_w_k(w, k);
                }
            }
        }
        // update N_z_k
        #pragma omp parallel for schedule(static) num_threads(_num_threads_)
        for (long k = 0; k < K; ++k) {
            for (long t = 0; t < num_batches_this_epoch; ++t) {
                N_z_k(k) = rho_phi * C / C_t[t] * N_hat_z_t_k(t, k) \
                    + (1 - rho_phi) * N_z_k(k);
            }
        }
    }
}

void calculate_topic() {
    if((topic = (struct _word_probability **) malloc(K * sizeof(struct _word_probability *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    for (long k = 0; k < K; ++k) {
        if((topic[k] = (struct _word_probability *) malloc(W * sizeof(struct _word_probability))) == NULL) {
            printf("Out of memory\n");
            exit(OUT_OF_MEMORY);
        }
        for (long i = 0; i < W; ++i) {
            topic[k][i].word = i + 1;
            topic[k][i].probability = phi_w_k(i + 1, k);
        }
    }

    // sort topic probabilities over words
    #pragma omp parallel for schedule(static) num_threads(2)
    for (long k = 0; k < K; ++k) {
        merge_sort(topic[k], W);
    }
}

void output() {
    FILE * output_file;

    // topics.txt
    if ((output_file = fopen("topics.txt", "w")) == NULL) {
        printf("Can't open topics.txt to write\n");
        exit(CANNOT_OPEN_FILE);
    }
    for (long k = 0; k < K; ++k) {
        for (long i = 0; i < NUM_TERMS_REPORTED_PER_TOPIC; ++i) {
            fprintf(output_file, "%ld:%8.6f", topic[k][i].word, topic[k][i].probability);
            if (i == NUM_TERMS_REPORTED_PER_TOPIC - 1) {
                fprintf(output_file, "\n");
            } else {
                fprintf(output_file, ", ");
            }
        }
    }
    if (0 != fclose(output_file)) {
        printf("Can't close topics.txt\n");
        exit(CANNOT_CLOSE_FILE);
    }

    // doctopic.txt
    if ((output_file = fopen("doctopic.txt", "w")) == NULL) {
        printf("Can't open doctopic.txt to write\n");
        exit(CANNOT_OPEN_FILE);
    }
    for (long d = 1; d <= D; ++d) {
        for (long k = 0; k < K; ++k) {
            fprintf(output_file, "%8.6f", theta_d_k(d, k));
            if (k == K - 1) {
                fprintf(output_file, "\n");
            } else {
                fprintf(output_file, ", ");
            }
        }
    }
    if (0 != fclose(output_file)) {
        printf("Can't close doctopic.txt\n");
        exit(CANNOT_CLOSE_FILE);
    }
}

