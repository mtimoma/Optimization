#include "Methods.h"

int count[7] = {0,0,0,0,0,0,0};

int * createList(int numStim){
	static int all_elements[numStim];
	for(int i=0;i<numStim;i++){
		all_elements[i] = i;
	}

	return all_elements;
}

int argmax(oracle,all_elements){
	int b;

	max_imp = -999999999

	for(int a=0;a<sizeof(all_elements)/sizeof(all_elements[0]);a++){
		//put in old 'evaluate' function in here
		Pr = oracle.Pr_h[a] //this doesn't actually work
		imp = entropy(Pr) //this too

		if(imp > max_imp){
			max_imp = imp;
			b = a
		}
	}
	return(b);
}

void update(oracle, int a){
	oracle.resp = neuron_resp(a);
	oracle.Ph = ;//Bayes;
	oracle.Phistory; //add the new Ph
}

float probSum(int a, int b){
	int difference = b - a;
	float sum = 0;

	for(int i=0;i<difference;i++){
		sum += oracle.Ph[a+i];
	}

	return(sum);
}

int findMax(int a, int b){
	int difference = b - a;
	int currentMax = 0;
	int currentMaxIndex = 0;
	for(int i=0;i<difference;i++){
		if(oracle.Ph[a+i] > currentMax){
			currentMaxIndex = a+i;
		}
	}

	return(currentMaxIndex);
}

int check(oracle){
	float prob_red = probSum(0,8);
	float prob_brown = probSum(8,16);
	float prob_blue = oracle.Ph[16];
	float prob_green = probSum(17,25);
	float prob_purple = probSum(25,33);
	float prob_gray = probSum(33,41);
	float prob_uncl = oracle.Ph[41];

	if(prob_red>0.999){
		count[0] += 1;
		rot = findMax(0,8);
	}
	if(prob_brown>0.999){
		count[1] += 1;
		rot = findMax(8,16);
	}
	if(prob_blue>0.999){
		count[2] += 1;
		rot = 16;
	}
	if(prob_green>0.999){
		count[3] += 1;
		rot = findMax(17,25);
	}
	if(prob_purple>0.999){
		count[4] += 1;
		rot = findMax(25,33);
	}
	if(prob_gray>0.999){
		count[5] += 1;
		rot = findMax(33,41);
	}
	if(prob_uncl>0.999){
		count[6] += 1;
	}
	else{
		//divide every number in count[] by 2
		//or, for stricter condition, reset to 0
		count[7] = {0,0,0,0,0,0,0};
	}

	for(i=0;i<7;i++){
		if(count[i]==10){
			return(i);
		}
	}
	return(100);
	//return a number greater than 7 so main knows
	//that there is no hypothesis
}