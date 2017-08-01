from __future__ import division
import numpy as np
from scipy.stats import pearsonr

#Abstract Class for Oracles

class Oracle(object):

    def evaluate(self,a):

        raise NotImplementedError()

    def update(self,a):

        global meanresult

        self.response = self.neuron_resp(a)

        self.Ph = self.Pr_h[a,self.response,:]*self.Ph/sum(self.Pr_h[a,self.response,:]*self.Ph)        
        self.Phistory.append(self.Ph)

        self.Pr_h = probh[means[self.mIndex]-1]
        self.Phm = meanresult[self.mIndex]

        probStim = 0.0

        for i in range(len(means)):
          for j in range(41):
            probStim += probh[means[i]-1][a][self.response][j] * self.Ph[j] * meanresult[i]

        self.Phm = (self.Pr_h[a,self.response,:].dot(self.Ph))*self.Phm/probStim
        meanresult[self.mIndex] = self.Phm

def argmax(oracle,all_elements):

    max_imp = -999999999

    for a in all_elements:

        imp = oracle.evaluate(a)

        if imp > max_imp:
            max_imp = imp
            b = a

    #print(b)

    oracle.update(b)

    return b


class greedy(object):

    def __init__(self,oracle,all_elements,k):

        self.oracle = oracle
        self.elements = all_elements
        self.k = k
        self.chosen_elements = []

    def solve(self):
        
        if self.elements == []:
            return []

        chosen = argmax(self.oracle,self.elements)

        return chosen


class Information_utility(Oracle):
    '''Mutual information Utility'''
    def __init__(self,Pr_h,Ph,neuron_resp,mIndex):
        self.Ph = Ph
        self.Pr_h = Pr_h
        self.neuron_resp = neuron_resp
        self.Phistory = [Ph]
        self.mIndex = mIndex
    
    def evaluate(self,a):
        Pr = self.Pr_h[a,:,:].dot(self.Ph)
        return entropy(Pr) - entropy(self.Pr_h[a,:,:]).dot(self.Ph)

        
class Uncertainty_utility(Oracle):
    '''Uncertainity Utility'''
    def __init__(self,Pr_h,Ph,neuron_resp,mIndex):
        self.Ph = Ph
        self.Pr_h = Pr_h
        self.neuron_resp = neuron_resp
        self.Phistory = [Ph]
        self.mIndex = mIndex

    def evaluate(self,a):
        Pr = self.Pr_h[a,:,:].dot(self.Ph)
        return entropy(Pr)


def entropy(P):
    return -sum(P * np.log(P))


def check(GC):
    global count
    global rot
       
    prob_red = sum(GC.oracle.Ph[0:8])
    prob_brown = sum(GC.oracle.Ph[8:16])
    prob_blue = GC.oracle.Ph[16]
    prob_green = sum(GC.oracle.Ph[17:25])
    prob_purple = sum(GC.oracle.Ph[25:33])
    prob_gray = sum(GC.oracle.Ph[33:41])

    if prob_red > 0.999:
        count[0] += 1
        rot = GC.oracle.Ph.tolist().index(max(GC.oracle.Ph[0:8]))
    elif prob_brown > 0.999:
        count[1] += 1
        rot = GC.oracle.Ph.tolist().index(max(GC.oracle.Ph[8:16]))
    elif prob_blue > 0.999:
        count[2] += 1
        rot = 16
    elif prob_green > 0.999:
        count[3] += 1
        rot = GC.oracle.Ph.tolist().index(max(GC.oracle.Ph[17:25]))
    elif prob_purple > 0.999:
        count[4] += 1
        rot = GC.oracle.Ph.tolist().index(max(GC.oracle.Ph[25:33]))
    elif prob_gray > 0.999:
        count[5] += 1
        rot = GC.oracle.Ph.tolist().index(max(GC.oracle.Ph[33:41]))
    else:
        count = [0,0,0,0,0,0]

    if 10 in count:
        return count.index(10)

def checkMeans(stimu):
    global mcount
    global means

    for prob in meanresult:
        if prob >= 0.8:
            mcount[meanresult.index(prob)] += 1

    if mcount == [0]*5 and stimu%50 == 0 and stimu != 0:
        nm = means[meanresult.index(max(meanresult))]
        mcount = [0]*5
        if nm in range(3,24):
            means = list(range(nm-2,nm+3))
        else:
            means = means

    if 10 in mcount:
        nm = means[mcount.index(10)]
        mcount = [0]*5
        if nm in range(3,24):
            means = list(range(nm-2,nm+3))
        else:
            means = means


def checkPearsonR(responses,rot,mIndex):
    expected = []
    for each in list(responses.keys()):
        expected.append(ideal[mIndex][rot][each])

    #print(pearsonr(expected,list(responses.values()))[0])
    return pearsonr(expected,list(responses.values()))[0]
    
def neuron_resp(a):
 
    np.random.seed(seednum)
    return np.random.poisson(50*X[neuronnum][a])


###################
# INITIALIZE DATA #
###################

# Load normalized responses
X = np.load('./data/all_resp.npy')

# Prob of Hypotheses?  Clustering
probh = np.load('./data/probh.npy')
ideal = np.load('./data/ideal.npy')

clusters = ['Red','Brown','Blue','Green','Purple','Gray']

#  ph - Array of current Pr[hypoth]
ph = np.array([.02]*16 + [0.2] + [.02]*24)

mf = open('MutualAll.csv','w')
uf = open('UncertAll.csv','w')

cells = int(input('Cells: '))
trials = int(input('Trials: '))

for i in range(trials):
    mf.write('Cluster,Stimuli,Size,')
for i in range(trials):
    uf.write('Cluster,Stimuli,Size,')

for n in range(cells):
    mf.write('\n')
    uf.write('\n')
    for s in range(trials):

        # which neurons to test
        neuronnum = n
        seednum = 1000 + 123*s
        ####################################
        # Create Objects and Run Simulation #
        ####################################

        # Mutual Information
        means = [3,8,13,18,23]
        meanresult = [0.2]*5
        GO = []
        for i in range(5):
            NC = Information_utility(probh[means[i]-1],ph,neuron_resp,i)
            GO.append(greedy(NC,list(range(362)),300))
        count , mcount = [0]*6 , [0]*5
        responses = {}
        rot = 0
        results = [None]*5
        done = False
        for i in range(300):
            stimShown = []
            for GC in GO:
                stimShown.append(GC.solve())

            for stim in range(len(stimShown)):
                responses[stimShown[stim]] = GO[stim].oracle.response

            if means[0] == means[1] - 1:
                for GC in GO:
                    checkC = check(GC)
                    if checkC != None:
                        if checkPearsonR(responses,rot,means[GO.index(GC)]-1) >= 0.6:
                            results[GO.index(GC)] = clusters[checkC]
                            done = True

            if i == 299 and checkC == None:
                mf.write('Unclassified,300,')
                mf.write(str(means[meanresult.index(max(meanresult))])+',')

            if done == True:
                if results[meanresult.index(max(meanresult))] is not None:
                    mf.write(results[meanresult.index(max(meanresult))] + ',')
                    mf.write(str(i) + ',')
                    mf.write(str(means[meanresult.index(max(meanresult))]) + ',')
                    break

            checkMeans(i)

        # Uncertainty
        means = [3,8,13,18,23]
        meanresult = [0.2]*5
        GO_Dual = []
        for i in range(5):
            NC_Dual = Information_utility(probh[means[i]-1],ph,neuron_resp,i)
            GO_Dual.append(greedy(NC,list(range(362)),300))
        count , mcount = [0]*6 , [0]*5
        responses = {}
        rot = 0
        results = [None]*5
        done = False
        for i in range(300):
            stimShown = []
            for GC in GO_Dual:
                stimShown.append(GC.solve())

            for stim in range(len(stimShown)):
                responses[stimShown[stim]] = GO_Dual[stim].oracle.response

            if means[0] == means[1] - 1:
                for GC in GO_Dual:
                    checkC = check(GC)
                    if checkC != None:
                        if checkPearsonR(responses,rot,means[GO_Dual.index(GC)]-1) >= 0.6:
                            results[GO.index(GC)] = clusters[checkC]
                            break

            if i == 299 and checkC == None:
                uf.write('Unclassified,300,')
                uf.write(str(means[meanresult.index(max(meanresult))])+',')

            if done == True:
                if results[meanresult.index(max(meanresult))] is not None:
                    uf.write(results[meanresult.index(max(meanresult))]+',')
                    uf.write(str(i)+',')
                    uf.write(str(means[meanresult.index(max(meanresult))])+',')
                    break

            checkMeans(i)

mf.close()
uf.close()
