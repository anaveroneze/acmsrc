#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include "fofaccomp.h"

#define debug false

//---------------------------------------------------------------------------
/**************************** FoF ******************************/
//---------------------------------------------------------------------------
#pragma acc routine seq
int friends_acc(float *x_bloco, float *y_bloco, float *z_bloco, int n, float rperc2, int* igru, int b){
  int u, j, l;
  float dist;
  int k = 0;

  #pragma acc data present(igru[0:n])
  for (u = 0; u < n; u++){
    // Group identifier - Denotates a new group
    if(igru[u] == 0){
      k++;
      igru[u] = k; //Find a new particle without group and start a group for it
      for (j = u ; j < n ; j++){ //Verifies the igru ids regarding particles not classified
        if(igru[j] == k){ //Verifies if it is a new group
          for (l = (u + 1) ; l < n ; l++){
            if (igru[l] == 0){ //If the group l is not defined:
              //Calculates the distance between particles: current(l) and previous(j)
              dist = (x_bloco[j] - x_bloco[l])*(x_bloco[j] - x_bloco[l]) + (y_bloco[j] - y_bloco[l])*(y_bloco[j] - y_bloco[l]) + (z_bloco[j] - z_bloco[l])*(z_bloco[j] - z_bloco[l]);
              if (dist <= rperc2) //If friends update the group id with k
                igru[l] = k;
            }
          }
        }
      }
    }
  }
  return k;
}

//---------------------------------------------------------------------------
/**************************** Post processing ******************************/
// Verify all particles and its groups
// Vector of groups + Current group id + New Group id + Number of elements +
// Number of blocks + Particles left over
//---------------------------------------------------------------------------
void group(int** igru, int thisgroup, int newgroup, int elem, int maxblocos, int leftover){

  int b, i;

  /*
    Parallelize the reclassification in CPU using OpenMP
  */

  for(b=0; b < maxblocos; b++){
  #pragma omp parallel for
	for(i = 0; i < elem; i++){
      if(igru[b][i] == thisgroup){
        igru[b][i] = newgroup;
      }
    }
  }

  for(i = 0; i < leftover; i++){
    if(igru[b][i] == thisgroup)
      igru[b][i] = newgroup;
  }
}

/******************** Comparation function used to sort *********************/
int compare(const void *a, const void *b){
  int xa = *(const int*) a;
  int xb = *(const int*) b;
  return (x[xa] > x[xb]) - (x[xa] < x[xb]);
}

//---------------------------------------------------------------------------
/************************** Friends-of-Friends *****************************/
//---------------------------------------------------------------------------
void fof(int b, float rperc){

  int i, j, resto, max, elem, numblocos;
  int *aux = malloc(sizeof(int)*N);
  int start, end, cont, numgrupos;

  qsort(id, N, sizeof(float), compare);

  resto = N%b;
  printf("Left over: %d\n", resto);
  max = b;
  printf("Max: %d\n", max);
  elem = N/max;

  /*The division between the total number of particles and the number of blocks may not
    be the same so we calculate the left over and create another block if needed
  */
  if(resto != 0){
    if(resto > elem){
      numblocos = resto/elem;
      resto = resto%elem;
      if(resto != 0) numblocos++;
      b += numblocos;
      max = b-1;
    }
    else
      b++;
  }

#if debug==true
  printf("All blocks: %d\nFull blocks: %d\nMax elements per block: %d\n", b, max, elem);
  printf("Left over: %d\n\n", resto);
#endif

  /************* Memory allocation ****************/

  float **x_bloco = (float **)malloc(sizeof(float*)*b);
  float **y_bloco = (float **)malloc(sizeof(float*)*b);
  float **z_bloco = (float **)malloc(sizeof(float*)*b);
  int **igru = (int **)malloc(sizeof(int*)*b);

  for(i=0; i<b; i++){
    x_bloco[i] = (float*)malloc(sizeof(float)*elem);
    y_bloco[i] = (float*)malloc(sizeof(float)*elem);
    z_bloco[i] = (float*)malloc(sizeof(float)*elem);
    igru[i] = (int*)malloc(sizeof(int)*elem);
    memset(igru[i], 0, elem*sizeof(int));
  }

  /************* Reads the vectors data ****************/
  numgrupos = 0;

  for(j=0; j<max; j++){
    start = (j*(N-resto))/max;
    end = ((j+1)*(N-resto))/max;
    cont = 0;
    for(i=start; i < end; i++, cont++){
      x_bloco[j][cont] = x[id[i]];
      y_bloco[j][cont] = y[id[i]];
      z_bloco[j][cont] = z[id[i]];
      aux[i] = id[i];
    }
  }

  //If it has left over
  for(i=end, cont=0; i<N; i++, cont++){
    x_bloco[j][cont] = x[id[i]];
    y_bloco[j][cont] = y[id[i]];
    z_bloco[j][cont] = z[id[i]];
  }

  /**************************** FoF on GPU ******************************/
  float rperc2 = rperc*rperc;
  //int ngpus = acc_get_num_devices(acc_device_nvidia);
  //printf("NGPUS: %d\n", ngpus);
  //acc_set_device_num(2, acc_device_nvidia);
  #pragma acc parallel loop pcopy(x_bloco[0:b][0:elem], y_bloco[0:b][0:elem], z_bloco[0:b][0:elem], igru[0:b][0:elem]) reduction(+:numgrupos) private(i)
  for(i = 0; i<max; i++){
    numgrupos += friends_acc(x_bloco[i], y_bloco[i], z_bloco[i], elem, rperc2, igru[i], i);
  }

  if(resto != 0)
    numgrupos += friends_acc(x_bloco[max], y_bloco[max], z_bloco[max], resto, rperc2, igru[max], max);

  printf("Groups: %d\n", numgrupos);

  /********* Print the groups *********************/
  start = 0;
  int MAX = 0;

  for(i=0;i<max;i++){
     //printf("\nBLOCO %d:\n", i);
     for(j=0;j<elem;j++){
       igru[i][j] = igru[i][j] + MAX;
       //printf("%d %d particula - grupo %d - x: %.2f y: %.2f z: %.2f\n", id[(elem*i)+j], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
       if(igru[i][j] > start)
          start = igru[i][j];
     }
     MAX = start;
  }

  //printf( "\nBLOCO RESTO %d\n", i);
  for(j=0;j<resto;j++){
    igru[i][j] = igru[i][j] + start;
    //printf("%d %d particula - grupo %d - x: %.2f y: %.2f z: %.2f\n", id[cont], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
  }

  float x_ant = x_bloco[0][elem-1];
  float y_ant = y_bloco[0][elem-1];
  float z_ant = z_bloco[0][elem-1];
  int igru_ant = igru[0][elem-1];
  float dist_x, dist_y, dist_z;
  int b1, b2, resto_calc;
  cont = 0;

  printf("\nPost-processing...\n");

  //For each block:
  for(b1 = 0; b1 < max; b1++){

    //For each block element (from the last to the first):
    for(i = elem - 1; i >= 0; i--){
      resto_calc = 1;
      //Compare a particle from current block with others:
      x_ant = x_bloco[b1][i];
      y_ant = y_bloco[b1][i];
      z_ant = z_bloco[b1][i];
      igru_ant = igru[b1][i];

      //Block
      for(b2 = b1+1; b2 < max; b2++){
        //Particle
        for(j = 0; j < elem; j++){
          //Verifies if the particles belongs to the same group
          if(igru[b2][j] != igru_ant){
            dist_x = (x_bloco[b2][j] - x_ant)*(x_bloco[b2][j] - x_ant);

            if(rperc2 >= dist_x) { //If the distance in coord X are less than R, continue
              dist_y = (y_bloco[b2][j] - y_ant)*(y_bloco[b2][j] - y_ant);
              dist_z = (z_bloco[b2][j] - z_ant)*(z_bloco[b2][j] - z_ant);

              //If the particles are friends:
              if( (rperc2 >= (dist_x + dist_y + dist_z)) ){
                  numgrupos--;
                  group(igru, igru[b2][j], igru_ant, elem, max, resto);
              }
            }else{
               j = elem;
               b2 = max;
               resto_calc = 0;
            }
          }
        }
      }

      if(resto_calc == 1){
        //Compare with particles of the left over block
        for(j=0; j<resto;j++){
            if(igru[b2][j] != igru_ant){ //If particles are from diferent groups compare again
                  dist_x = (x_bloco[b2][j] - x_ant)*(x_bloco[b2][j] - x_ant);
                  if(rperc2 >= dist_x){
                        dist_y = (y_bloco[b2][j] - y_ant)*(y_bloco[b2][j] - y_ant);
                        dist_z = (z_bloco[b2][j] - z_ant)*(z_bloco[b2][j] - z_ant);
                        if( (rperc2 >= (dist_x + dist_y + dist_z)) ){
                            numgrupos--;
                            group(igru, igru[b2][j], igru_ant, elem, max, resto);
                        }
                  }
            }
        }
      }
    }
  }

#if debug==true
  for(i=0;i<max;i++){
     printf("\nBLOCK %d:\n", i);
     for(j=0;j<elem;j++){
       printf("%d %d particle - group %d - x: %.2f y: %.2f z: %.2f\n", id[(elem*i)+j], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
     }
  }

  printf( "\nLEFT BLOCK %d\n", i);
  for(j=0;j<resto;j++){
    printf("%d %d particle - group %d - x: %.2f y: %.2f z: %.2f\n", id[cont], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
  }
#endif

  printf("Groups after post-processing: %d\n", numgrupos);

/**************************** Clean the memory ******************************/
  for(i=0; i<b; i++){
    free(x_bloco[i]);
    free(y_bloco[i]);
    free(z_bloco[i]);
    free(igru[i]);
  }

  free(x_bloco);
  free(y_bloco);
  free(z_bloco);
  free(aux);
  free(igru);
}

