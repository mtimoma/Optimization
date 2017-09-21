int numStim;
//number of stimuli available
int *stim;
//list of stimuli shown
int *responses;
//list of responses
int a;
//next stimulus to be shown
int rot;
int clus;
int count[7];

int * createList(int numStim);

typedef struct{
	float Ph[42]; //probabilities of pypotheses with fixed size 42
	float Pr_h[362][80][42]; //probability of responses
	int resp; //response based on poisson of data
	float * Phistory; //store past Ph in dynamic list
}oracle;

void update(int a);

int argmax(oracle,all_elements);
//method used to choose next stimulus using mutual information

float entropy(P);
//find entropy unless we find a math library containing this
//parameter P is an array of probabilities

int check(oracle);
//checks if any probability of hypothesis >0.999
//returns index of highest probable hypothesis

float checkPearsonR(*stim,*responses,int rot,int clus);
//calculates PearsonR between current cell and cores of same cluster
//add a library?

int neuron_resp(int a);
//returns response of neuron based on poisson of data

float findMean(*stim,*responses);
//finds means for each hypothesis based on data

float sum(A);
//sum an array unless we find a math library

float probSum(int a, int b);
//sums probabilities from index a to index b

int findMax(int a, int b);
//finds maximum probability from index a to b