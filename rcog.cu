/*
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
 -- Date Started: 	07-26-2016
 -- Date Completed: 08-24-2016
 -- Program title: 	RCOG (Radio Frequency Interferencce Characterization on GUPPI)
 -- Purpose:
	This program is the overhead for RCOG (RFI Characterization On Guppi). RCOG will take guppi raw data and produce a spectrogram that has been flagged through statistical analysis. The reason for RCOG is to compare the RCOG flagged data to unflagged data to see if this program is capable of mitigating RFI before it is averaged for pulsar science.
	This will take the data and pass half of the channels to one GPU and the other half of the channels to the other GPU. Each GPU will then calculate statistics and perform an FFT. The FFT will produce a spectrogram that is then written to file. The statistics will be calculated with the same time resolution of as the FFT and will be written to a separate file. All channels will have separate files that are written to. This means that there will be two output files for all channels.
 -- Input: The infile must be of GUPPI raw data format. It must be read from disk file, not streamed. 
 -- Output: There are 16 outfiles. All of which are in binary.
*/

#include <fcntl.h> // for pipes
#include <unistd.h> // for pipes
#include <stdlib.h> // for pipes
#include <sys/stat.h> // for pipes
#include <stdio.h>
#include <memory.h>
#include <cuda.h> 
#include <cuComplex.h>
#include "rcog_read_gupp.h"
#include "rcog_spectro.h" // global variables here
#include "rcog_stat.h"
#define samp_T0 65536 // how many 160 ns samples in period of .01048576 seconds
#define CH_BYTES 33548288
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
	/*************************************************************************************************/
	/********************************************* Main **********************************************/
	/*************************************************************************************************/

int main(int argc, char *argv[]){

	float ftime = 0;
	double dtime = 0;
	static cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	FILE *fp_in;
	FILE *fp_stat[8];
	FILE *fp_fft[8];
	int bytes_leftover = 0;
	char filename[88];
	int chan;
	int usePipes = 0; // for pipes
	int n1, n2, p1, p2; // for pipes

	/*************************/
	/*** Command Line Arg ****/
	/*************************/

	if (argc >= 2) {sscanf(argv[1], "%s", filename);}
	else {
		printf("FATAL: main(): command line argument <filename> not specified\n");
	  	return 0;
	}
	printf("filename = '%s'\n",filename);	
	if (argc>=3) {sscanf( argv[2], "%d", &chan );}  // make chan is 25.
	else { 
		printf("FATAL: main(): command line argument <chan> not specified\n");
		return 0;
	}
	if ( chan > 25){ printf("Pick a channel less than 25 thanks!"); return 0;}
	printf("channels = %d through %d\n", chan, chan+7);
	if (argc>=4) {sscanf( argv[3], "%d", &usePipes);} // hidden parameter

	// OUT FILES . . .
	if (!(fp_in = fopen(filename,"r"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}
	fp_stat[0] = fopen("stat0.dat","w");
	fp_stat[1] = fopen("stat1.dat","w");
	fp_stat[2] = fopen("stat2.dat","w");
	fp_stat[3] = fopen("stat3.dat","w");
	fp_stat[4] = fopen("stat4.dat","w");
	fp_stat[5] = fopen("stat5.dat","w");
	fp_stat[6] = fopen("stat6.dat","w");
	fp_stat[7] = fopen("stat7.dat","w");
	fp_fft[0] = fopen("spectro0.dat","w");
	fp_fft[1] = fopen("spectro1.dat","w");
	fp_fft[2] = fopen("spectro2.dat","w");
	fp_fft[3] = fopen("spectro3.dat","w");
	fp_fft[4] = fopen("spectro4.dat","w");
	fp_fft[5] = fopen("spectro5.dat","w");
	fp_fft[6] = fopen("spectro6.dat","w");
	fp_fft[7] = fopen("spectro7.dat","w");

	if (usePipes == 1){
		// open fifo for write
		if((p1 = open("specHermes", O_WRONLY)) < 0) {printf( "p1 did not open boss!\n"); return -2;} // for pipes
		else{printf("p1WRONLY opened successfully!\n");}	 // for pipes
		if((p2 = open("statHermes", O_WRONLY)) < 0) {printf( "p2 did not open boss!\n"); return -2;} // for pipes
		else{printf("p2WRONLY opened successfully!\n");}	 // for pipes
	}

	/***************************************/
	/************* Stats Memory ************/
	/***************************************/

	char4 *h_input;
	char4 *d_input; 
	float2 *d_m2; 
	float2 *d_me;
	int2 *d_ma;
	int2 *d_mi;
	float2 *d_resizer; 	
	int2 *d_histo; 
	float2 *d_mom;
	float2 *h_me;
	int2 *h_ma;
	struct rsk *h_rsk;
	struct rsk *d_rsk;

	// Host Allocation (page-locked)
	CUDA_CALL(cudaHostAlloc((void**) &h_input, channels*nT0*samp_T0*sizeof(char4), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_me, channels*nT0*sizeof(float2), cudaHostAllocDefault)); 
	CUDA_CALL(cudaHostAlloc((void**) &h_ma, channels*nT0*sizeof(int2), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_rsk, channels*nT0*sizeof(struct rsk), cudaHostAllocDefault));

	// Device Allocation 
	CUDA_CALL(cudaMalloc((void**)&d_input, channels*nT0*samp_T0*sizeof(char4)));
	CUDA_CALL(cudaMalloc((void**)&d_m2, channels*nT0*samp_T0*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**)&d_me, channels*nT0*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**)&d_ma, channels*nT0*sizeof(int2)));
	CUDA_CALL(cudaMalloc((void**)&d_mi, channels*nT0*sizeof(int2)));
	CUDA_CALL(cudaMalloc((void**)&d_resizer, channels*nT0*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**)&d_histo, channels*nT0*hist_size*sizeof(int2)));
	CUDA_CALL(cudaMalloc((void**)&d_mom, channels*nT0*nMom*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**)&d_rsk, channels*nT0*sizeof(struct rsk)));
    
	/***************************************/
	/************** FFT Memory *************/
	/***************************************/
	
    float2 *d_fft_mem;
	float *d_avg_pwr;
	float *h_avg_pwr;

	// Host Allocation (page-locked)
	CUDA_CALL(cudaHostAlloc((void**)&h_avg_pwr, channels*samps_out*sizeof(float), cudaHostAllocDefault));

	// Device Allocation
	CUDA_CALL(cudaMalloc((void**) &d_fft_mem, channels*samps_in*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**) &d_avg_pwr, channels*samps_out*sizeof(float)));

	/*************************/
	/****** Main Loop ********/
	/*************************/

	int mm = 0;
	while(!feof(fp_in)){
		cudaEventRecord(start);
		// reformat data
		printf("\nRead raw guppi file, section: %d\n", mm);
		bytes_leftover = Rcog_read_gupp(fp_in, chan, h_input, bytes_leftover, mm);
		if (bytes_leftover < 1){break;}
		printf("Read successful! \n");
		Rcog_stat(h_input, d_input, d_m2, d_me, d_ma, d_mi, d_resizer, d_histo,
					 d_mom, h_me, h_ma, h_rsk, d_rsk, fp_stat);
		Rcog_spectro(d_input, d_fft_mem, d_avg_pwr, h_avg_pwr, fp_fft);
		
		// pipe data
		if (usePipes == 1){
			printf("ALL: writing to pipe\n");
			for (int res = 0; res < nT0; res++){ // 5 for channel 25 + 5 = 30
				n1 += write(p1, &h_avg_pwr[(5*samps_out)+(fft_size*res)], fft_size*sizeof(float));
				n2 += write(p2, &h_me[5*nT0+res], sizeof(float));
				n2 += write(p2, &h_rsk[5*nT0+res].rms, sizeof(float));
				n2 += write(p2, &h_rsk[5*nT0+res].skew, sizeof(float));
				n2 += write(p2, &h_rsk[5*nT0+res].kurt, sizeof(float));
			}
			printf("FFT bytes passed : %d\nSTAT bytes passed : %d\n", n1, n2);
			n1 = 0; n2 = 0;
		}

		// should ask what the time resolution should be or something. This will change the averaging.
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime( &ftime, start, stop);
		dtime += (ftime/1000);
		mm++;
	}

	printf("\nMain: Program Ran for: %e s\n", dtime);

	// cleanup memory
	cudaFree(d_input);
	cudaFree(d_m2);
	cudaFree(d_me);
	cudaFree(d_ma);   
	cudaFree(d_mi);
	cudaFree(d_resizer);
	cudaFree(d_histo);
	cudaFree(d_mom);
	cudaFree(d_rsk);
	cudaFree(d_fft_mem);
	cudaFree(d_avg_pwr);
	cudaFreeHost(h_input);
	cudaFreeHost(h_me);
	cudaFreeHost(h_ma);
	cudaFreeHost(h_rsk);
	cudaFreeHost(h_avg_pwr);

	fclose(fp_in);
	if (usePipes == 1){
		close(p1);
		close(p2);
	}
	for (int ch = 0; ch < channels; ch++){
		fclose(fp_stat[ch]);
		fclose(fp_fft[ch]);
	}
	return 0;
}

// to run: rcog /lustre/pulsar/users/rprestag/1713+0747_global/raw/guppi_56465_J1713+0747_0006.0000.raw 25
/********************/
/* CUDA ERROR CHECK */
/********************/
/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
*/
