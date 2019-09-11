#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include "fofacc.h"

//---------------------------------------------------------------------------
/**************************** FoF ******************************/
//---------------------------------------------------------------------------
#pragma acc routine seq
int FriendsAcc(float *x_bloco, float *y_bloco, float *z_bloco, int n, float rperc2, int* igru, int b){
  int u, j, l;
  float dist;
  int k = 0;

  #pragma acc data present(igru[0:n])
  for (u = 0; u < n; u++){
    // Identificador do grupo - Denomina um novo grupo
    if(igru[u] == 0){
      k++;
      igru[u] = k; //Encontra uma partícula sem grupo e começa um grupo para ela
      for (j = u ; j < n ; j++){ //Verifica os indices de igru a partir da partícula sem grupos
        if(igru[j] == k){ //Verifica se é um grupo ainda não definido
          for (l = (u + 1) ; l < n ; l++){ //Então percorre o resto do vetor de grupos
            if (igru[l] == 0){ //Se o grupo na posição l ainda não foi definido:
              //Calcula a distância entre a partícula atual(l) e a anterior(j)
              dist = (x_bloco[j] - x_bloco[l])*(x_bloco[j] - x_bloco[l]) + (y_bloco[j] - y_bloco[l])*(y_bloco[j] - y_bloco[l]) + (z_bloco[j] - z_bloco[l])*(z_bloco[j] - z_bloco[l]);
              if (dist <= rperc2) //Se encontrou partículas amigas seta o grupo para k
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
/**************************** FoF ******************************/
// Percorre todas as partículas modificando os seus grupos
// Vetor de grupos; Grupo antes; Grupo atual; Grupo novo; Nº elementos;
//---------------------------------------------------------------------------
void seta_grupos(int*igru1, int*igru2, int grupoatual, int gruponovo, int elem){

  int i;
  for(i = 0; i < elem; i++){
    if(igru1[i] == grupoatual)
      igru1[i] = gruponovo;

    if(igru2[i] == grupoatual)
        igru2[i] = gruponovo;
  }
}

//TENTATIVA PARALELO
// void seta_grupos(int** igru, int grupoatual, int gruponovo, int elem, int maxblocos, int resto){
//
//   int b, i;
//
//   #pragma acc data create(igru[0:maxblocos][0:elem])
//   {
//     #pragma acc update device(igru[0:maxblocos][0:elem])
//     for(b=0; b < maxblocos; b++){
//       #pragma acc parallel loop
//       for(i = 0; i < elem; i++){
//         if(igru[b][i] == grupoatual){
//           igru[b][i] = gruponovo;
//         }
//       }
//     }
//     #pragma acc update self(igru[0:maxblocos][0:elem])
//   }
//
//   for(i = 0; i < resto; i++){
//     if(igru[b][i] == grupoatual)
//       igru[b][i] = gruponovo;
//   }
// }

int compara(const void *a, const void *b){
  int xa = *(const int*) a;
  int xb = *(const int*) b;
  return (x[xa] > x[xb]) - (x[xa] < x[xb]);
}

//---------------------------------------------------------------------------
/**************************** Chamada do FoF ******************************/
//---------------------------------------------------------------------------
void fof(int b, float rperc){

  int i, j, resto, max, elem, m, numblocos;
  int *aux = malloc(sizeof(int)*N);
  int start, end, cont, numgrupos;

  qsort(id, N, sizeof(float), compara);

  /*************Aloca memória para leitura dos dados****************/
  resto = N%b;
  printf("\nResto inicial: %d \n", resto);
  max = b;
  elem = N/max;

  if(resto != 0){//A divisão do número total de partículas pelo numero de blocos não é exata
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

  printf("Blocos sem resto: %d\nBlocos total: %d\nElem máx por bloco: %d\n", max, b, elem);
  printf("Resto real: %d\n\n", resto);

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

  /*************Lendo dados dos vetores já preenchidos****************/
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

  //Lê o resto se tiver
  for(i=end, cont=0; i<N; i++, cont++){
    x_bloco[j][cont] = x[id[i]];
    y_bloco[j][cont] = y[id[i]];
    z_bloco[j][cont] = z[id[i]];
  }

  /**************************** FoF on GPU ******************************/
  float rperc2 = rperc*rperc;

  #pragma acc parallel loop pcopy(x_bloco[0:b][0:elem], y_bloco[0:b][0:elem], z_bloco[0:b][0:elem], igru[0:b][0:elem]) reduction(+:numgrupos) private(i)
  for(i = 0; i<max; i++){
    numgrupos += FriendsAcc(x_bloco[i], y_bloco[i], z_bloco[i], elem, rperc2, igru[i], i);
  }

  if(resto != 0)
    numgrupos += FriendsAcc(x_bloco[max], y_bloco[max], z_bloco[max], resto, rperc2, igru[max], max);

  printf("Grupos encontrados: %d\n", numgrupos);

  /********* Imprime os grupos encontrados *********************/
  start = 0;
  int MAX = 0;

  for(i=0;i<max;i++){
  //   printf("\nBLOCO %d:\n", i);
     for(j=0;j<elem;j++){
       igru[i][j] = igru[i][j] + MAX;
    //   printf("%d %d particula - grupo %d - x: %.2f y: %.2f z: %.2f\n", id[(elem*i)+j], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
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
  int k, bloco, particula, resto_m, c;
  int prox = 0;
  cont = 0;
  int novo = -1;
  int change = 0;

  printf("\n\nPós-processamento:\n");

  int b1, b2, resto_calc; //Variáveis para percorrer os blocos

  //Para cada bloco:
  for(b1 = 0; b1 < max; b1++){

    //Para cada elemento no bloco (percorrendo do último para o primeiro):
    for(i = elem - 1; i >= 0; i--){
      resto_calc = 1;
      //Compara uma partícula do bloco atual com as dos outros blocos:
      x_ant = x_bloco[b1][i];
      y_ant = y_bloco[b1][i];
      z_ant = z_bloco[b1][i];
      igru_ant = igru[b1][i];

      //Bloco a ser comparado:
      for(b2 = b1+1; b2 < max; b2++){
        //Partículas do bloco comparado:
        for(j = 0; j < elem; j++){
          //Verifica se as partículas são de grupos distintos
          if(igru[b2][j] != igru_ant){
            dist_x = (x_bloco[b2][j] - x_ant)*(x_bloco[b2][j] - x_ant);

            if(rperc2 >= dist_x) { //Se a distância entre as partículas sobre o eixo X for menor que o raio, continua
              dist_y = (y_bloco[b2][j] - y_ant)*(y_bloco[b2][j] - y_ant);
              dist_z = (z_bloco[b2][j] - z_ant)*(z_bloco[b2][j] - z_ant);

              //Se as partículas pertencem ao mesmo grupo:
              if( (rperc2 >= (dist_x + dist_y + dist_z)) ){
                //printf("rperc: %f >=  (dist_x + dist_y + dist_z) %f\n\n", rperc2, dist_x, dist_y, dist_z);
                //printf("GRUPO ANTES: %d (bloco: %d part: %d)    GRUPO DEPOIS: %d\n\n", igru[b2][j], b2, j, igru_ant);
                change++;
                //Troca pro grupo de menor valor
                if(igru[b2][j] > igru_ant)
                  //seta_grupos(igru, igru[b2][j], igru_ant, elem, max, resto);
                  seta_grupos(igru[b2], igru[b1], igru[b2][j], igru_ant, elem);
                else
                  //seta_grupos(igru, igru_ant, igru[b2][j], elem, max, resto);
                  seta_grupos(igru[b2], igru[b1], igru_ant, igru[b2][j], elem);
              }

            }else{ //Senão as próximas partículas não precisam ser verificadas (pois está ordenado por X)
               j = elem;
               b2 = max;
               resto_calc = 0;
            }
          }
        }
      }

      if(resto_calc == 1){
        //Compara com as partículas do bloco de resto
        for(j=0; j<resto;j++){
            if(igru[b2][j] != igru_ant){ //se as partículas sao de grupos diferentes compara para ver se nao junta
                  dist_x = (x_bloco[b2][j] - x_ant)*(x_bloco[b2][j] - x_ant);
                  if(rperc2 >= dist_x){
                        dist_y = (y_bloco[b2][j] - y_ant)*(y_bloco[b2][j] - y_ant);
                        dist_z = (z_bloco[b2][j] - z_ant)*(z_bloco[b2][j] - z_ant);
                        if( (rperc2 >= (dist_x + dist_y + dist_z)) ){
                          //printf("GRUPO ANTES: %d (bloco: %d part: %d)    GRUPO DEPOIS: %d\n\n", igru[b2][j], b2, j, igru_ant);
                          //numgrupos--;
                          change++;
                          if(igru[b2][j] > igru_ant)
                            //seta_grupos(igru, igru[b2][j], igru_ant, elem, max, resto);
                            seta_grupos(igru[b2], igru[b1], igru[b2][j], igru_ant, resto);
                          else
                            //seta_grupos(igru, igru_ant, igru[b2][j], elem, max, resto);
                            seta_grupos(igru[b2], igru[b1], igru_ant, igru[b2][j], resto);
                        }
                  }
            }
        }
      }
    }
  }

  //printf("Grupos pós processamento: %d\n", numgrupos);
  printf("Mudancas: %d\n", change);

  // for(i=0;i<max;i++){
  //    printf("\nBLOCO %d:\n", i);
  //    for(j=0;j<elem;j++){
  //      printf("%d %d particula - grupo %d - x: %.2f y: %.2f z: %.2f\n", id[(elem*i)+j], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
  //    }
  // }
  //
  // printf( "\nBLOCO RESTO %d\n", i);
  // for(j=0;j<resto;j++){
  //   printf("%d %d particula - grupo %d - x: %.2f y: %.2f z: %.2f\n", id[cont], j, igru[i][j], x_bloco[i][j], y_bloco[i][j], z_bloco[i][j]);
  // }

/**************************** Limpa a memória ******************************/
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
