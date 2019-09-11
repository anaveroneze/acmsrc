/***************************************************************************
Programa que realiza o agrupamento de partículas de acordo com o raio de
percolação fornecido, usando o algoritmo Friends of Friends para um ambiente
de computação híbrida (GPU+CPU) com OpenACC.
Ultima atualização:
Autor: Ana Luisa Veroneze Solórzano
$pgcc -g fofacc.c -o run -ta=nvidia:nvidia -Minfo
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
/********************* Cálculo do tempo em usec *****************/
//---------------------------------------------------------------------------
long getTime(){
  struct timeval time;
  gettimeofday(&time, (struct timezone *) NULL);
  return time.tv_sec*1000000 + time.tv_usec;
}

//---------------------------------------------------------------------------
/****************** Leitura dos dados e chamada do FoF *********************/
//---------------------------------------------------------------------------
void le_dados(char *filename, int b, float rperc){

  int i, *aux;

  FILE  *file = fopen(filename,"rt");
  fscanf (file, "%d", &N); //Número total de partículas

  x  = malloc(sizeof(float)*N);  // posição x da partícula;
  y  = malloc(sizeof(float)*N);  // posição y da partícula;
  z  = malloc(sizeof(float)*N);  // posição z da partícula
  v1 = malloc(sizeof(float)*N);  // coordenada x da velocidade
  v2 = malloc(sizeof(float)*N);  // coordenada y da velocidade
  v3 = malloc(sizeof(float)*N);  // coordenada z da velocidade
  id = malloc(sizeof(int)*N);
  aux = malloc(sizeof(int)*N);

  /*************Leitura dos dados do arquivo e ordenação pelo eixo x ****************/
  for(i = 0; i<N; i++){
    fscanf (file, "%d %f %f %f %f %f %f", &aux[i], &x[i], &y[i], &z[i], &v1[i], &v2[i], &v3[i]);
    id[i] = i;
  }

  fclose(file);
}

//---------------------------------------------------------------------------
/**************************** Limpa a memória ******************************/
//---------------------------------------------------------------------------
void limpa_memo(void){
  free(x);
  free(y);
  free(z);
  free(v1);
  free(v2);
  free(v3);
}

//---------------------------------------------------------------------------
/***************************************************************************/
//---------------------------------------------------------------------------
int main(int argc, char **argv){
  float  local_v[100];
  int num_threads;
  char *arg1;
  long start_fof, stop_fof, start_read, stop_read, start_novo, stop_novo;

  if(argc != 4 ){
    puts( "Por favor, entre com o nome do arquivo de dados, raio de percolação e nº de blocos." );
    exit(1);
  }

  arg1 = argv[1];
  float rperc = atof(argv[2]);
  num_threads = atoi(argv[3]);

  puts ("Iniciando...\n");

  start_read = getTime();
  le_dados(arg1, num_threads, rperc);
  stop_read = getTime();
  printf("\nTempo Leitura: %ld\n", (long)(stop_read - start_read));

  start_fof = getTime();
  fof(num_threads, rperc);
  stop_fof = getTime();
  
  printf("\nTempo FoF: %ld\n", (long)(stop_fof - start_fof));
  printf("--------------------\n");
  limpa_memo();
  return 0;
}
