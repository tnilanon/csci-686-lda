#ifndef FAST_LDA_UTIL_H_
#define FAST_LDA_UTIL_H_

void read_sparse_dataset(char input_file_name[]);

void start_timer();
void stop_timer(char message[]);

struct _word_probability
{
    long word;
    double probability;
};

void merge_sort(struct _word_probability topic_dist[], long n);

#endif  // FAST_LDA_UTIL_H_

