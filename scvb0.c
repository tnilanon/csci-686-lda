#include <stdio.h>
#include <stdlib.h>
#include "error_code.h"
#include "dataset.h"

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
    num_topics = strtoul(argv[3], NULL, 10);
    if (num_topics == 0) {
        printf("Recheck number of topics\n");
        exit(INVALID_NUM_TOPICS);
    }

    read_sparse_dataset(argv[1]);

    printf("first (%lu): %lu %lu\n", size_i[1], word_i_j[1][0], count_i_j[1][0]);
    printf("last (%lu): %lu %lu\n", size_i[D], word_i_j[D][size_i[D] - 1], count_i_j[D][size_i[D] - 1]);
}

