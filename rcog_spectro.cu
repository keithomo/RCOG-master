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

 -- Author: 		Keith Omogrosso
 -- Institution: 	National Radio Astronomy Observatory (Green Bank)
 -- Program: 		RCOG
 -- Date Completed:	08-24-2016
 -- Purpose: 		To Perform the FFT for 8 channels for 512 time chunks. It can do no more based on GPU memory limitations. It will then write to disk in 8 separate binary files; one binary file for each channel. Each time rcog_spectro() runs it will append a new 512 time chunks for each of 8 channels to each of 8 files currently open.
 -- Input: 			Prior to calling rcog_spectro(), the buffer of d_input needs to be filled with current data. This is done in rcog_stat. Logically then, rcog_stat() needs to be called before rcog_spectro(). The outfiles also need to be opened. This happens in main.
 -- Output:			After rcog_spectro(), spectrogram for 512 time chunks will have been written to file. 
******************************************************************************************************/
#include <cuda.h>
#include <stdio.h>
#include <memory.h>
#include <cufft.h>
#include <cuComplex.h>
#include "rcog_spectro.h" // global variables here
#define CUFFT_CALL(call) \
do { \
    cufftResult status = call; \
    if (status != CUFFT_SUCCESS) \
    { \
        printf("cufft Error %d on line %d\n", status, __LINE__); \
    } \
    } while(0)

				/***************************************/
				/******** Function Definitions *********/
				/***************************************/

__global__
void Convert_to_floatX(char4 *in, float2 *out)
{// ignore z and w portions because that is y polarization not x
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[idx].x = (float)in[idx].x;
    out[idx].y = (float)in[idx].y;
}

__global__ 
void Power_arr(float2 *in, float2 *out)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	// calculate X(s)X(s)*  = power
    out[idx].x = in[idx].x * in[idx].x + in[idx].y * in[idx].y;
	out[idx].y = 0;
}

// you can speed up this I believe by combining both power_arr and avg_to_ms

__global__
void Avg_to_ms(float2 *d_fft_mem, float *d_avg_pwr) // make sure grd is recalculated
{
	// This uses two different indexes because the two arrays are of different size.
	int idxo  = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxi  = (blockIdx.x * blockDim.x * avg_size) + threadIdx.x;
	d_avg_pwr[idxo] = 0; // just in case 
	for (int ii = 0; ii < fft_size*avg_size; ii+= fft_size){
		d_avg_pwr[idxo] += d_fft_mem[((idxi) + ii)].x; 
	}
	d_avg_pwr[idxo] = d_avg_pwr[idxo]/avg_size;
}

int Simple_fft(float2 *in, float2 *out, cufftHandle plan)
{ 
	if (cufftExecC2C(plan, in, out, CUFFT_FORWARD) != CUFFT_SUCCESS){ 
		printf("CUFFT error: ExecC2C Forward Failed\n");
		return -1;
	}
	if (cudaDeviceSynchronize() != cudaSuccess){
		printf("Cuda error: Failed to synchronize\n");
		return -1;
	}
	return 0;
}

				/***************************************/
				/**************** Main *****************/
				/***************************************/

int Rcog_spectro(
		char4 *d_input,
		float2 *d_fft_mem,
		float *d_avg_pwr,
		float *h_avg_pwr,
		FILE *fp_fft[8]){

	// variables
	dim3 blk(fft_size); // must be fft_size!
    dim3 grd(samps_in/fft_size); // batch
	dim3 grd_avg(samps_out/fft_size);
	int a = samps_in;
	int b = samps_out;

	// time keeps
	float fft_time = 0;
	static cudaEvent_t start_fft, stop_fft;
	cudaEventCreate(&start_fft);
	cudaEventCreate(&stop_fft);
	cudaEventRecord(start_fft);
	cufftHandle plan;
	CUFFT_CALL(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch));

					/*************************/
					/* initialize /main loop */
					/*************************/

		// CUDA START // NO MEMCPY NEEDED // STAT_KING ALREADY DID THAT //
	for (int ch = 0; ch < channels; ch++){
		Convert_to_floatX<<<grd,blk>>>(&d_input[ch*a], &d_fft_mem[ch*a]);   
		Simple_fft(&d_fft_mem[ch*a], &d_fft_mem[ch*a], plan);
		Power_arr<<<grd,blk>>>(&d_fft_mem[ch*a], &d_fft_mem[ch*a]);
		Avg_to_ms<<<grd_avg,blk>>>(&d_fft_mem[ch*a], &d_avg_pwr[ch*b]);
	}
	for (int ch = 0; ch < channels; ch++){
		for (int jj = 0; jj < (samps_out/fft_size); jj++){ // flip the bandstop to bandpass. out of 512
			cudaMemcpy(&h_avg_pwr[ch*b + (fft_size*jj)], &d_avg_pwr[ch*b + half_BW + (fft_size*jj)], half_BW*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_avg_pwr[ch*b + half_BW + (fft_size*jj)], &d_avg_pwr[ch*b + (fft_size*jj)], half_BW*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}	// CUDA END // SPECTROGRAM PRODUCED // WRITE TO FILE AND RETURN //
	
	// write array to binary
	for (int ch = 0; ch < channels; ch++){
		fwrite(&h_avg_pwr[ch*b], samps_out*sizeof(float), 1, fp_fft[ch]);
	}
	
 	// clean-up
	cufftDestroy(plan);
	cudaEventRecord(stop_fft);
	cudaEventSynchronize(stop_fft);
	cudaEventElapsedTime( &fft_time, start_fft, stop_fft);
	printf("\n	FFT: All of it took: %3.3f ms\n", fft_time);
	cudaEventDestroy(start_fft);
	cudaEventDestroy(stop_fft);

   	return 0;
}

/*************** FOR ERROR CHECKING INLINE *************
cudaError_t cudaError;
cudaError = cudaGetLastError();
if(cudaError != cudaSuccess)
{
	printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	return 0;
}
********************************************************/

/************************************************************************************
Input: 		
Output:
Purpose:
*************************************************************************************/

