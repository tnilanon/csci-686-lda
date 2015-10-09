#include "error_code.h"
#include "dataset.h"

unsigned long * word_temp, * count_temp;

void read_sparse_dataset(char input_file_name[]) {

    FILE *input_file = fopen(input_file_name, "r");
    if (input_file == NULL) {
        printf("Can't open docword.txt\n");
        exit(CANNOT_OPEN_FILE);
    }

    fscanf(input_file, "%lu\n%lu\n%lu\n", &D, &W, &NNZ);
    printf("D = %lu; W = %lu; NNZ = %lu;\n", D, W, NNZ);
    N = 0;

    if ((word_temp = (unsigned long *) malloc(W * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((count_temp = (unsigned long *) malloc(W * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((size_d = (unsigned long *) malloc((D + 1) * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((word_d_i = (unsigned long **) malloc((D + 1) * sizeof(unsigned long *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((count_d_i = (unsigned long **) malloc((D + 1) * sizeof(unsigned long *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    
    long next_temp, this_doc, last_doc = -1;
    for (long i = 0; i < NNZ; ++i) {
        if (i % 100000 == 0 && i != 0) {
            printf("Done reading %ld lines\n", i);
        }
        fscanf(input_file, "%lu ", &this_doc);
        // New document
        if (this_doc != last_doc) {
            if (last_doc != -1) {
                // Malloc and copy over
                size_d[last_doc] = next_temp;
                long copy_size = next_temp * sizeof(unsigned long);
                if ((word_d_i[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
                    printf("Out of memory\n");
                    exit(OUT_OF_MEMORY);
                }
                memcpy(word_d_i[last_doc], word_temp, copy_size);
                if ((count_d_i[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
                    printf("Out of memory\n");
                    exit(OUT_OF_MEMORY);
                }
                memcpy(count_d_i[last_doc], count_temp, copy_size);
            }
            last_doc = this_doc;
            // Reset temp array
            next_temp = 0;
        }
        fscanf(input_file, "%lu %lu\n", &word_temp[next_temp], &count_temp[next_temp]);
        N += count_temp[next_temp];
        ++next_temp;
    }
    size_d[last_doc] = next_temp;
    long copy_size = next_temp * sizeof(unsigned long);
    if ((word_d_i[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memcpy(word_d_i[last_doc], word_temp, copy_size);
    if ((count_d_i[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memcpy(count_d_i[last_doc], count_temp, copy_size);

    printf("N = %lu;\n", N);
}

