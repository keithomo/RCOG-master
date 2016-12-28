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
 -- Purpose: 		To read raw guppi formatted data from disk. This does not work for streaming on the guppi system in real time. It does this for 8 CONSECUTIVE channels for 512 time chunks. It can do no more based on GPU memory limitations. It will read the guppi data into the buffer. Each time rcog_spectro() runs it will append a new 512 time chunks for each of 8 channels to each of 8 files currently open.
 -- Input: 			Prior to calling rcog_spectro(), the buffer of d_input needs to be filled with current data. This is done in rcog_stat. Logically then, rcog_stat() needs to be called before rcog_spectro(). The outfiles also need to be opened. This happens in main.
 -- Output:			After rcog_spectro(), spectrogram for 512 time chunks will have been written to file. 
******************************************************************************************************/
#include<stdio.h>
#include <memory.h>
#include <stdlib.h>
#include "rcog_spectro.h" // global variables here

#define BPFPC 4293918720 //bytes per file per channel not including overlap
#define BLOCKSIZE 1073545216
#define CH_BYTES 33548288 // including overlap
#define OVERLAPB 2048 // bytes of overlap
#define NPOL 4 // bytes per 160 ns sample
#define HEADER_BYTES 6240

// memory for h_input gets overwritten every time this file runs from main loop. 
// You are altering rcog, read_gupp, and stat_king to input nT0*channels*sizeof(data)*amount_of_data into your alloc. Then try out your stat code!

int Rcog_read_gupp(FILE *fp_in, int chan, char4 *h_input, int bytes_leftover, int mm)
{
	int nBlks;
	int trans_s = (CH_BYTES -OVERLAPB); // transfer size in bytes
	int proc_buff = samp_T0*nT0*NPOL; // 134,217,728 bytes per channel
	int bytes_movefirst = 0;
	float npasses = ((float)((float)BPFPC) / ((float)proc_buff)); // yeah overkill to make it a float :P
	int N = nT0*samp_T0; // # of main loops needed to exhaust file
	int nt0;

	/***********************/
	/***** Calculation *****/
	/***********************/	

	if ( npasses < mm+1){ // make sure you have enough groceries to cook the meal. last time only
		nt0 = (int)(((float)npasses - (float)mm)*nT0);
		proc_buff = samp_T0*nt0*NPOL;
	}
	if(bytes_leftover > 0){	bytes_movefirst = trans_s - bytes_leftover;}
	proc_buff = proc_buff - bytes_movefirst;
	nBlks = (int)floor((proc_buff)/trans_s);
	bytes_leftover = (proc_buff) - (nBlks*trans_s);
	if ( npasses < mm+1){printf("bytes at the end: %d\n", bytes_leftover);} 

	/*************************/
	/**** read guppi data ****/
	/*************************/

	// eat the left overs (meal 0 you don't have left overs cause there was no meal -1!)
	if(bytes_movefirst > 0){ 
		fseek(fp_in, HEADER_BYTES, SEEK_CUR);
		fseek(fp_in, CH_BYTES*(chan-1), SEEK_CUR); // beginning of chan
		for (int ww = 0; ww < channels; ww++){
			fseek(fp_in, OVERLAPB + (trans_s-bytes_movefirst), SEEK_CUR); // where we left off
			fread( &h_input[ww*N+0], bytes_movefirst*sizeof(char), 1, fp_in);
		}
		// go to end of block
		fseek(fp_in,CH_BYTES*(32-chan-channels+1), SEEK_CUR);
	}

	// Hungry (or still hungry) so you make and new food
	for (int ii = 0; ii < nBlks; ii++){ 
		fseek(fp_in, HEADER_BYTES, SEEK_CUR);
		fseek(fp_in, CH_BYTES*(chan-1), SEEK_CUR); // beginning of chan
		for (int ww = 0; ww < channels; ww++){
			fseek(fp_in, OVERLAPB, SEEK_CUR); // after overlap
			fread( &h_input[ww*N+((bytes_movefirst+(ii*trans_s))/NPOL)], trans_s*sizeof(char), 1, fp_in);
		}
		// go to end of block
		fseek(fp_in, CH_BYTES*(32-chan-channels+1), SEEK_CUR);
	}
	
	// leave left overs in the fridge
	fseek(fp_in, HEADER_BYTES, SEEK_CUR);
	fseek(fp_in, CH_BYTES*(chan-1), SEEK_CUR); // beginning of chan
	for (int ww = 0; ww < channels; ww++){
		fseek(fp_in, OVERLAPB, SEEK_CUR); // after overlap
		fread( &h_input[ww*N+((bytes_movefirst+(nBlks*trans_s))/NPOL)], bytes_leftover*sizeof(char), 1, fp_in);
		if (ww < (channels-1)){fseek(fp_in, (trans_s-bytes_leftover), SEEK_CUR);} // beginning of next channel
	}
	// go back to beginning of block nBlk+1
	fseek(fp_in, (trans_s-bytes_leftover)+CH_BYTES*(32-chan-channels+1)-(BLOCKSIZE + HEADER_BYTES) , SEEK_CUR); 
// confusing math ^^^ but it is actually rewinding to the beginning of the left overs block		
// I did it this way to avoid any EOF mistakes with loosing the last bit of data. 
	
	return bytes_leftover;
}
