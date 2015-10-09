#ifndef FAST_LDA_DATASET_H_
#define FAST_LDA_DATASET_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

unsigned long num_iterations, K;
unsigned long D, W, NNZ;
unsigned long * size_i, ** word_i_j, ** count_i_j;

void read_sparse_dataset(char input_file_name[]);

#endif  // FAST_LDA_DATASET_H_

