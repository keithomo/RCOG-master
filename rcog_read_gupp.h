#ifndef rcog_read_gupp_h
#define rcog_read_gupp_h
#include <stdio.h>
#include <cuda.h>


int Rcog_read_gupp(FILE *fp_in, int chan, char4 *h_input, int bytes_leftover, int mm);
                
#endif
