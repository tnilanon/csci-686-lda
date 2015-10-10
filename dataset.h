#ifndef FAST_LDA_DATASET_H_
#define FAST_LDA_DATASET_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

long num_iterations, K;
long D, W, NNZ, C;
long * size_d, * C_d, ** word_d_i, ** count_d_i;

void read_sparse_dataset(char input_file_name[]);

#endif  // FAST_LDA_DATASET_H_

