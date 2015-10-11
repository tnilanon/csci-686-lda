#include "error_code.h"
#include <math.h>
#include <sys/time.h>

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

void start_timer() {
    gettimeofday(&tic, NULL);
}

void stop_timer(char message[]) {
    gettimeofday(&toc, NULL);
    timeval_subtract(&diff, &toc, &tic);
    printf("\t");
    printf(message, diff.tv_sec + (double) diff.tv_usec / 1000000);
}

