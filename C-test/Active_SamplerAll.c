#include <stdio.h>
#include <stdlib.h>
#include "Methods.h"

//load all the arrays

//initialize starting probabilities
float ph[42] = {
	0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
	0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004,
	0.14,
	0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,
	0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,
	0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,
	0.692
}

int initStim[5];
srand(0);
for(int i=0;i<5;i++){
	initStim[i] = rand()%362;
	//random number out of 361; likely incorrect
}
int initResponse[5];
for(int i=0;i<5;i++){
	initResponse[i] = neuron_resp(initStim[i])
}