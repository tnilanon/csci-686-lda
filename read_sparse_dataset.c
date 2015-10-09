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
    printf("D = %lu; W = %lu; NNZ = %lu\n", D, W, NNZ);

    if ((word_temp = (unsigned long *) malloc(W * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((count_temp = (unsigned long *) malloc(W * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((size_i = (unsigned long *) malloc((D + 1) * sizeof(unsigned long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((word_i_j = (unsigned long **) malloc((D + 1) * sizeof(unsigned long *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((count_i_j = (unsigned long **) malloc((D + 1) * sizeof(unsigned long *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    
    int next_temp, this_doc, last_doc = -1;
    for (int i = 0; i < NNZ; ++i) {
        if (i % 100000 == 0 && i != 0) {
            printf("Lines done: %d\n", i);
        }
        fscanf(input_file, "%lu ", &this_doc);
        // printf("this_doc: %d; ", this_doc);
        // New document
        if (this_doc != last_doc) {
            if (last_doc != -1) {
                // Malloc and copy over
                size_i[last_doc] = next_temp;
                int copy_size = next_temp * sizeof(unsigned long);
                if ((word_i_j[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
                    printf("Out of memory\n");
                    exit(OUT_OF_MEMORY);
                }
                memcpy(word_i_j[last_doc], word_temp, copy_size);
                if ((count_i_j[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
                    printf("Out of memory\n");
                    exit(OUT_OF_MEMORY);
                }
                memcpy(count_i_j[last_doc], count_temp, copy_size);
                last_doc = this_doc;
                // printf("%lu\n", size_i[1]);
                // for (int j = 0; j < size_i[1]; ++j) {
                //     printf("%lu %lu => %lu %lu\n", word_temp[j], count_temp[j], word_i_j[1][j], count_i_j[1][j]);
                // }
            }
            else {
                last_doc = this_doc;
            }
            // Reset temp array
            next_temp = 0;
        }
        fscanf(input_file, "%lu %lu\n", &word_temp[next_temp], &count_temp[next_temp]);
        // printf("word: %d; count: %d; temp: %d\n", word_temp[next_temp], count_temp[next_temp], next_temp);
        ++next_temp;
    }
    last_doc = this_doc;
    size_i[last_doc] = next_temp;
    int copy_size = next_temp * sizeof(unsigned long);
    if ((word_i_j[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memcpy(word_i_j[last_doc], word_temp, copy_size);
    if ((count_i_j[last_doc] = (unsigned long *) malloc(copy_size)) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memcpy(count_i_j[last_doc], count_temp, copy_size);
}

