/***************************************************************************
Program to classify particles from dark matter simulations according to its
percolation radius using a Friends-of-Friends algorithm for a hybrid computing
environment (CPU+GPU) using OpenACC.
Author: Ana Luisa Veroneze Solórzano
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include "fofaccomp.h"

float *x, *y, *z, *v1, *v2, *v3, value;
int N, *id;

//---------------------------------------------------------------------------
/********************* Execution time in usec *****************/
//---------------------------------------------------------------------------
long getTime(){
  struct timeval time;
  gettimeofday(&time, (struct timezone *) NULL);
  return time.tv_sec*1000000 + time.tv_usec;
}

//---------------------------------------------------------------------------
/****************** Read the input data *********************/
//---------------------------------------------------------------------------
void read_data(char *filename, int b, float rperc){

  int i, *aux;

  FILE  *file = fopen(filename,"rt");
  fscanf (file, "%d", &N); //Get total number of particles

  //Get the particles information
  x  = malloc(sizeof(float)*N);  // position x;
  y  = malloc(sizeof(float)*N);  // position y;
  z  = malloc(sizeof(float)*N);  // position z;
  v1 = malloc(sizeof(float)*N);  // velocity coordinate x
  v2 = malloc(sizeof(float)*N);  // velocity coordinate y
  v3 = malloc(sizeof(float)*N);  // velocity coordinate z
  id = malloc(sizeof(int)*N);
  aux = malloc(sizeof(int)*N);

  for(i = 0; i<N; i++){
    fscanf (file, "%d %f %f %f %f %f %f", &aux[i], &x[i], &y[i], &z[i], &v1[i], &v2[i], &v3[i]);
    id[i] = i;
  }

  fclose(file);
}

//---------------------------------------------------------------------------
/********************** Clean the memory allocated **************************/
//---------------------------------------------------------------------------
void clean_memo(void){
  free(x);
  free(y);
  free(z);
  free(v1);
  free(v2);
  free(v3);
}

//---------------------------------------------------------------------------
/*********************** MAIN FUNCTION HERE ****************************/
//---------------------------------------------------------------------------
int main(int argc, char **argv){

  float  local_v[100];
  int num_threads;
  char *arg1;
  long start_fof, stop_fof, start_read, stop_read, start_novo, stop_novo;

  if(argc != 4 ){
    puts( "Please, enter with the input file name, the percolation radius and the number of blocks." );
    exit(1);
  }

  arg1 = argv[1];
  float rperc = atof(argv[2]);
  num_threads = atoi(argv[3]);

  puts ("Starting...\n");

  start_read = getTime();
  read_data(arg1, num_threads, rperc);
  stop_read = getTime();
  printf("\nReading time: %ld\n", (long)(stop_read - start_read));

  start_fof = getTime();
  fof(num_threads, rperc);
  stop_fof = getTime();

  printf("\nFoF time: %ld\n", (long)(stop_fof - start_fof));
  printf("--------------------\n");
  clean_memo();
  return 0;
}
