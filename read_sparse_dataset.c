#include "error_code.h"
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

long * word_temp, * count_temp;

void read_sparse_dataset(char input_file_name[]) {

    FILE *input_file = fopen(input_file_name, "r");
    if (input_file == NULL) {
        printf("Can't open docword.txt to read\n");
        exit(CANNOT_OPEN_FILE);
    }

    if (3 != fscanf(input_file, "%ld\n%ld\n%ld\n", &D, &W, &NNZ)) {
        printf("There is something wrong with input file format\n");
        exit(INVALID_INPUT_FILE_FORMAT);
    }
    printf("D = %ld; W = %ld; NNZ = %ld;\n", D, W, NNZ);

    C = 0;
    count_max = 0;

    if ((word_temp = (long *) malloc(W * sizeof(long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((count_temp = (long *) malloc(W * sizeof(long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((size_d = (long *) malloc((D + 1) * sizeof(long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    if ((C_d = (long *) malloc((D + 1) * sizeof(long))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memset(C_d, 0, (D + 1) * sizeof(long));
    if ((word_d_i = (long **) malloc((D + 1) * sizeof(long *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    if ((count_d_i = (long **) malloc((D + 1) * sizeof(long *))) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    };
    
    printf("lines done reading: ");
    long next_temp, this_doc, last_doc = -1;
    for (long i = 0; i < NNZ; ++i) {
        if (i % 1000000 == 0 && i != 0) {
            printf("%ldM ", i/1000000);
            fflush(stdout);
        }
        if (1 != fscanf(input_file, "%ld ", &this_doc)) {
            printf("There is something wrong with input file format\n");
            exit(INVALID_INPUT_FILE_FORMAT);
        }
        // new document
        if (this_doc != last_doc) {
            if (last_doc != -1) {
                // malloc and copy over
                size_d[last_doc] = next_temp;
                long copy_size = next_temp * sizeof(long);
                if ((word_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
                    printf("Out of memory\n");
                    exit(OUT_OF_MEMORY);
                }
                memcpy(word_d_i[last_doc], word_temp, copy_size);
                if ((count_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
                    printf("Out of memory\n");
                    exit(OUT_OF_MEMORY);
                }
                memcpy(count_d_i[last_doc], count_temp, copy_size);
            }
            last_doc = this_doc;
            // reset temp array
            next_temp = 0;
        }
        if (2 != fscanf(input_file, "%ld %ld\n", &word_temp[next_temp], &count_temp[next_temp])) {
            printf("There is something wrong with input file format\n");
            exit(INVALID_INPUT_FILE_FORMAT);
        }
        long ctnt = count_temp[next_temp];
        C_d[this_doc] += ctnt;
        C += ctnt;
        count_max = (ctnt > count_max)? ctnt: count_max;
        ++next_temp;
    }
    size_d[last_doc] = next_temp;
    long copy_size = next_temp * sizeof(long);
    if ((word_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memcpy(word_d_i[last_doc], word_temp, copy_size);
    if ((count_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
        printf("Out of memory\n");
        exit(OUT_OF_MEMORY);
    }
    memcpy(count_d_i[last_doc], count_temp, copy_size);
    printf("\n");

    if (0 != fclose(input_file)) {
        printf("Can't close docword.txt\n");
        exit(CANNOT_CLOSE_FILE);
    }

    printf("C = %ld;\n", C);
}

