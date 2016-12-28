/*****************************************************************************************************
 Copyright (C) 2016 Keith Omogrosso
 Developed as part of the GBT project
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 
 Correspondence concerning GBT software should be addressed as follows:
       GBT Operations
       National Radio Astronomy Observatory
       P. O. Box 2
       Green Bank, WV 24944-0002 USA

 -- Author:			Keith Omogrosso
 -- Institution:	National Radio Astronomy Observatory (Green Bank)
 -- Program:		RCOG
 -- Date Completed:	08-24-2016
 -- Purpose:		To calculate statistics of mean, rms, skewness, and normal kurtosis on 512 time chunks over 8 frequency channels. It can do no more based on GPU memory limitations. It will then write to disk in 8 separate binary files; one binary file for each channel. Each time rcog_stat() runs it will append a new 512 time chunks for each of 8 channels to each of 8 files currently open.
 -- Input:			Prior to calling rcog_stat(), the buffer of h_input needs to be filled with current data. This is done in rcog_read_gupp. Logically then, rcog_read_gupp() needs to be called before rcog_stat(). The outfiles also need to be opened. This happens in main.
 -- Output:			After rcog_stat(), statistics for 512 time chunks will have been written to file. 
******************************************************************************************************/
#include <cuda.h> 
#include <stdio.h>
#include <memory.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdlib.h>
#include <iostream>
#include "rcog_stat.h"
#include "rcog_spectro.h" // global variables here
#define NPOL 4
#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", \
                 __FILE__, __LINE__, cudaGetErrorString(err) ); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUFFT_CALL(call) \
do { \
    cufftResult status = call; \
    if (status != CUFFT_SUCCESS) \
    { \
        printf("cufft Error %d on line %d\n", status, __LINE__); \
    } \
    } while(0)

// (1 / 6)
__global__
void Power_arr(char4 *in, float2 *m2) // already very fast. Cannot change that.
{ 
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	// calculate real^2 + imaginary^2  = power
    m2[idx].x = ((float)in[idx].x) * ((float)in[idx].x) + ((float)in[idx].y) * ((float)in[idx].y);
    m2[idx].y = ((float)in[idx].z) * ((float)in[idx].z) + ((float)in[idx].w) * ((float)in[idx].w);
}

// (2 / 6)
__global__
void Me_npol(float2 *m2, float2 *me) // slowest kernel
{ 
	float mx = 0, my = 0;
	int idx = threadIdx.x * blockDim.x + blockIdx.x * samp_T0;
	for (int ii = 0; ii < samp_T0/blockDim.x; ii++){
		mx += m2[idx + ii].x;
		my += m2[idx + ii].y;
	}
	__syncthreads();
	atomicAdd( &me[blockIdx.x].x, mx);
	atomicAdd( &me[blockIdx.x].y, my);
	__syncthreads();
	if (idx == blockIdx.x*samp_T0){
		me[blockIdx.x].x /= (samp_T0); 
		me[blockIdx.x].y /= (samp_T0); 
	}
}

// (3 / 6)
__global__
void Max_min(float2 *m2, int2 *ma, int2 *mi, float2 *resizer) // slowest kernel
{ 
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x) * 256;
	int threadMx = 0;
	int threadMy = 0;
	int threadmx = 0;
	int threadmy = 0;
	for (int ii = 0; ii < 256; ii++){
		if (threadMx < m2[idx + ii].x){threadMx = (int)m2[idx + ii].x;}
		if (threadMy < m2[idx + ii].y){threadMy = (int)m2[idx + ii].y;}
		if (threadmx > m2[idx + ii].x){threadmx = (int)m2[idx + ii].x;}
		if (threadmy > m2[idx + ii].y){threadmy = (int)m2[idx + ii].y;}
	}
	__syncthreads();
	atomicMax( &ma[blockIdx.x].x, threadMx);
	atomicMax( &ma[blockIdx.x].y, threadMy);
	atomicMin( &mi[blockIdx.x].x, threadmx);
	atomicMin( &mi[blockIdx.x].y, threadmy);
	__syncthreads();
	if (idx == blockIdx.x*blockDim.x*256){ //1.001 just to make division work later
		resizer[blockIdx.x].x = ((float)(ma[blockIdx.x].x - mi[blockIdx.x].x)/hist_size)*1.001; 
		resizer[blockIdx.x].y = ((float)(ma[blockIdx.x].y - mi[blockIdx.x].y)/hist_size)*1.001; 
	}
}

// (4 / 6)
__global__  
void Histo_kernel( float2 *m2, int2 *histo, int2 *mi, float2 *resizer){ // speed up improvements can be made here.
	__shared__ int2 tempH[hist_size];
	tempH[threadIdx.x] = make_int2(0., 0.);
	__syncthreads();
	int idx = threadIdx.x + (blockIdx.x*blockDim.x);
	int jump = floorf((blockIdx.x*blockDim.x)/samp_T0); // what time sample each threadblock is in.
	for (int zz = 0; zz < nT0; zz++){
		if (jump == zz){
			atomicAdd(&tempH[(int)((m2[idx].x-mi[zz].x)/resizer[zz].x )].x,1);
			atomicAdd(&tempH[(int)((m2[idx].y-mi[zz].y)/resizer[zz].y )].y,1); 
			__syncthreads();
			atomicAdd( &(histo[threadIdx.x + zz*hist_size].x), tempH[threadIdx.x].x);
			atomicAdd( &(histo[threadIdx.x + zz*hist_size].y), tempH[threadIdx.x].y);
		}
	}
}

// (5 / 6)
__global__ 
void Mom_order(float2 *me, int2 *histo, float2 *mom, int2 *mi, float2 *resizer)
{
	int next = blockIdx.x;
	int order = threadIdx.x;
	for (int ii = 0; ii < hist_size; ii++){
		mom[order+next*3].x += ((float)( powf( ((ii+mi[next].x)*resizer[next].x) - me[next].x, order+2)) * histo[ii+next*hist_size].x); 
		mom[order+next*3].y += ((float)( powf( ((ii+mi[next].y)*resizer[next].y) - me[next].y, order+2)) * histo[ii+next*hist_size].y); 
	}
	mom[order+next*3].x /= (samp_T0);
	mom[order+next*3].y /= (samp_T0);
}

// (6 / 6)
__global__
void Final(float2 *mom, struct rsk *d_rsk)
{
	int idx = blockIdx.x; // <<<512 "timeseries", 6 "made faster">>>
	if(threadIdx.x==0){ d_rsk[idx].rms.x = sqrtf(fabsf(mom[idx*3+0].x));}
	if(threadIdx.x==1){ d_rsk[idx].rms.y = sqrtf(fabsf(mom[idx*3+0].y));}
	if(threadIdx.x==2){ d_rsk[idx].skew.x = mom[idx*3+1].x/(powf(mom[idx*3+0].x, 1.5));}
	if(threadIdx.x==3){ d_rsk[idx].skew.y = mom[idx*3+1].y/(powf(mom[idx*3+0].y, 1.5));}
	if(threadIdx.x==4){ d_rsk[idx].kurt.x = (mom[idx*3+2].x/(powf(mom[idx*3+0].x, 2))) -3;}
	if(threadIdx.x==5){ d_rsk[idx].kurt.y = (mom[idx*3+2].y/(powf(mom[idx*3+0].y, 2))) -3;}
}

			/********************************************************/
			/********************* rcog_stat() **********************/
			/********************************************************/

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
		FILE *fp_stat[8]){	

	int outlet = 2;
	int a = nT0*samp_T0;
	int b = nT0;
	int c = nT0*hist_size;
	int d = nT0*nMom;
	cudaStream_t stream[channels];

	dim3 pwrgrd(nT0*samp_T0/hist_size);
	dim3 hist(hist_size);
	dim3 nto(nT0);
	dim3 nmom(nMom);

	// time keeps
	float elapsedTime=0, elapsedTimeK=0, elapsedTimeM=0, elapsedTimeO=0;
	static cudaEvent_t start_mem, stop_mem;
	static cudaEvent_t start_ker, stop_ker;
	static cudaEvent_t start_o, stop_o;
	cudaEventCreate(&start_mem);
	cudaEventCreate(&stop_mem);
	cudaEventCreate(&start_ker);
	cudaEventCreate(&stop_ker);
	cudaEventCreate(&start_o);
	cudaEventCreate(&stop_o);
	cudaEventRecord(start_mem);

				/************************************************/
				/********************  Loop  ********************/
				/************************************************/

	cudaMemset(d_me, 0, channels*nT0*sizeof(float2));
	cudaMemset(d_ma, 0, channels*nT0*sizeof(int2));
	cudaMemset(d_mi, 0, channels*nT0*sizeof(int2));
	cudaMemset(d_resizer, 0, channels*nT0*sizeof(float2));
	cudaMemset(d_histo, 0, channels*nT0*hist_size*sizeof(int2));
	cudaMemset(d_mom, 0, channels*nT0*nMom*sizeof(float2));
	cudaMemset(d_rsk, 0, channels*nT0*sizeof(struct rsk));

	cudaEventRecord(start_ker);
	for (int ch = 0; ch < channels; ch++){
		CUDA_CALL(cudaStreamCreate( &stream[ch]));
		CUDA_CALL(cudaMemcpyAsync(&d_input[ch*a+0], &h_input[ch*a+0], nT0*samp_T0*sizeof(char4), cudaMemcpyHostToDevice, stream[ch]));
		Power_arr<<<pwrgrd,hist, 0, stream[ch]>>>(&d_input[ch*a+0], &d_m2[ch*a+0]);
		Me_npol<<<nto,hist, 0, stream[ch]>>>(&d_m2[ch*a+0], &d_me[ch*b+0]);
		Max_min<<<nto,hist, 0, stream[ch]>>>(&d_m2[ch*a+0], &d_ma[ch*b+0], &d_mi[ch*b+0], &d_resizer[ch*b+0]);
		Histo_kernel<<<pwrgrd,hist, 0, stream[ch]>>>(&d_m2[ch*a+0], &d_histo[ch*c+0], &d_mi[ch*b+0], &d_resizer[ch*b+0]); 
		Mom_order<<<nto,nmom, 0, stream[ch]>>>(&d_me[ch*b+0], &d_histo[ch*c+0], &d_mom[ch*d+0], &d_mi[ch*b+0], &d_resizer[ch*b+0]);
		Final<<<nto,6, 0, stream[ch]>>>(&d_mom[ch*d+0], &d_rsk[ch*b+0]);
	}
	for (int ch = 0; ch < channels; ch++){
		cudaStreamSynchronize( stream[ch]);
		cudaMemcpyAsync(&h_me[ch*b+0], &d_me[ch*b+0], nT0*sizeof(float2), cudaMemcpyDeviceToHost, stream[ch]);
		cudaMemcpyAsync(&h_rsk[ch*b+0], &d_rsk[ch*b+0], nT0*sizeof(struct rsk), cudaMemcpyDeviceToHost, stream[ch]);
	}
	cudaDeviceSynchronize(); 
	cudaEventRecord(stop_ker);
	cudaEventSynchronize(stop_ker);
	cudaEventRecord(start_o);


				/************************************************/
				/*************  Write To Files  *****************/
				/************************************************/

	if (outlet == 1){ // Diagnostics
		cudaMemcpy(h_ma, d_ma, channels*nT0*sizeof(float2), cudaMemcpyDeviceToHost);
		int val = 0;
		for (int ch = 0; ch < 8; ch++){ // for each channel
			for (int next = val; next < (val+3); next++){ // for sequential values
				printf("For Channel %d:::\n", ch);
				printf("\nStat Sample: %d;\n", next);
				printf("M.x = %f; ", h_me[ch*b+next].x);
				printf("M.y = %f; \n", h_me[ch*b+next].y); 
				printf("^.x = %d; ",h_ma[ch*b+next].x);
				printf("^.y = %d; \n",h_ma[ch*b+next].y);
				printf("R.x = %f; ", h_rsk[ch*b+next].rms.x);
				printf("R.y = %f; \n", h_rsk[ch*b+next].rms.y);
				printf("S.x = %f; ", h_rsk[ch*b+next].skew.x);
				printf("S.y = %f; \n", h_rsk[ch*b+next].skew.y);
				printf("K.x = %f; ", h_rsk[ch*b+next].kurt.x);
				printf("K.y = %f; \n", h_rsk[ch*b+next].kurt.y);
			}
		}
	}
	if(outlet == 2){ // write to file
		for (int ch = 0; ch < channels; ch++){
			for (int ii = 0; ii < nT0; ii++){
				fwrite(&h_me[ch*b+ii], sizeof(float2), 1, fp_stat[ch]);
				fwrite(&h_rsk[ch*b+ii], sizeof(struct rsk), 1, fp_stat[ch]);
			}
		}
	}

	// Timing addition
	cudaEventRecord(stop_o, 0);
	cudaEventSynchronize(stop_o);
	cudaEventRecord(stop_mem, 0);
	cudaEventSynchronize(stop_mem);
	cudaEventElapsedTime( &elapsedTime, start_ker, stop_ker);
	elapsedTimeK += elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start_o, stop_o);
	elapsedTimeO += elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start_mem, stop_mem);
	elapsedTimeM += elapsedTime;
	printf("\n	STAT: Kernel calls took: %3.3f ms\n", elapsedTimeK);
	printf("	STAT: Writing to files and calculations took: %3.3f ms\n", elapsedTimeO);
	printf("	STAT: All Memory transfers took: %3.3f ms\n", elapsedTimeM - (elapsedTimeK + elapsedTimeO));
	printf("	STAT: All of stats took: %3.3f ms\n", elapsedTimeM);

	// Cleaning up
	cudaEventDestroy(start_mem);
	cudaEventDestroy(stop_mem);
	cudaEventDestroy(start_ker);
	cudaEventDestroy(stop_ker);
	cudaEventDestroy(start_o);
	cudaEventDestroy(stop_o);
	for (int ii = 0; ii < channels; ii++) cudaStreamDestroy(stream[ii]);
	return 0;
}
