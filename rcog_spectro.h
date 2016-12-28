#ifndef rcog_spectro_h
#define rcog_spectro_h
#include <stdio.h>

const int samp_T0 = 65536; // how many 160 ns samples in period of .01048576 seconds
const int fft_size = 1024; // cannot be more than 1024 // if you change, change for spectro too!
const int nT0 = 256;  	// if you change, change for other files too! Works from 128 to 512 well
const int avg_size = samp_T0/fft_size;			// fft time samples (avg_size*128) per loop
const int samps_in = nT0*samp_T0; // one picture
const int samps_out = samps_in/avg_size;
const int nMom = 3; // do not change
const int hist_size = 256; // do not change
const int half_BW = fft_size/2;				// 512
const int batch = samps_in/fft_size; 		// number of times FFT performed.
const int channels = 8; 

int Rcog_spectro(
		char4 *d_input,
		float2 *d_fft_mem,
		float *d_avg_pwr,
		float *h_avg_pwr,
		FILE *fp_fft[8]);
                
#endif  
