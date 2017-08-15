import numpy as np

def neuron_resp(a):

    return np.random.poisson(50*X[0][a])

X = np.load('./data/all_resp.npy')
Pr_h = np.load('./data/probh50.npy')


np.random.seed(1000)
Ph = [0.02]*41
for i in range(0,5,1):
    response = neuron_resp(i)
    Ph = Pr_h[i,response,:]*Ph/sum(Pr_h[i,response,:]*Ph)
print(Ph)

np.random.seed(1000)
Ph = [0.02]*41
for i in range(4,-1,-1):
    response = neuron_resp(i)
    Ph = Pr_h[i,response,:]*Ph/sum(Pr_h[i,response,:]*Ph)
print(Ph)