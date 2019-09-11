#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

extern float *x, *y, *z, *v1, *v2, *v3, value;
extern int N, *id;

int friends_acc(float *x_bloco, float *y_bloco, float *z_bloco, int n, float rperc2, int* igru, int b);
void group(int** igru, int grupoatual, int gruponovo, int elem, int maxblocos, int resto);
void fof(int b, float rperc);


#endif
