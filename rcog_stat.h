#ifndef rcog_stat_h
#define rcog_stat_h
#include <stdio.h>
#include <cuda.h> 

struct rsk {
	float2 rms;
	float2 skew;
	float2 kurt;
};

int Rcog_stat(
		char4 *h_input,
		char4 *d_input,
		float2 *d_m2,
		float2 *d_me,
		int2 *d_ma,
		int2 *d_mi,
		float2 *d_resizer,	
		int2 *d_histo,
		float2 *d_mom,
		float2 *h_me,
		int2 *h_ma,
		struct rsk *h_rsk,
		struct rsk *d_rsk,
		FILE *fp_stat[8]);

#endif
