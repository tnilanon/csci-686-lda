#include "error_code.h"
#include "dataset.h"

long * word_temp, * count_temp;

void read_sparse_dataset(char input_file_name[]) {

    FILE *input_file = fopen(input_file_name, "r");
    if (input_file == NULL) {
        printf("Can't open docword.txt\n");
        exit(CANNOT_OPEN_FILE);
    }

    fscanf(input_file, "%ld\n%ld\n%ld\n", &D, &W, &NNZ);
    printf("D = %ld; W = %ld; NNZ = %ld;\n", D, W, NNZ);

    C = 0;
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
    
    long next_temp, this_doc, last_doc = -1;
    for (long i = 0; i < NNZ; ++i) {
        if (i % 1000000 == 0 && i != 0) {
            printf("Done reading %ld lines\n", i);
        }
        fscanf(input_file, "%ld ", &this_doc);
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
        fscanf(input_file, "%ld %ld\n", &word_temp[next_temp], &count_temp[next_temp]);
        C_d[this_doc] += count_temp[next_temp];
        C += count_temp[next_temp];
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

    printf("C = %ld;\n", C);
}

