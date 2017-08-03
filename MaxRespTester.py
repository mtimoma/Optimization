import numpy as np


def neuron_resp(a):
    np.random.seed(seednum)
    return np.random.poisson(50*X[neuronnum-1][a])

X = np.load('./data/all_resp.npy')
# Prob of Hypotheses?  Clustering
probh = np.load('./data/probh50.npy')
ideal = np.load('./data/ideal50.npy')
norms = np.load('./data/maxr.npy')

# which neurons to test
neuronnum = int(input('Neuron: '))
seednum = int(input('Seed: '))

np.random.seed(seednum)
initStim = np.random.randint(362, size=5)

meanresp = np.array([0]*41)
for stim in initStim:
    for i in range(41):
        meanresp[i] += neuron_resp(stim) / norms[i][stim]
meanresp = meanresp / len(initStim)
meanresp = np.round(meanresp).astype(int)

print(meanresp)