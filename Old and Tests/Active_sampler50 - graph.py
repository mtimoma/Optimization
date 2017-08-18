from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
#from scipy.stats import pearsonr

#Abstract Class for Oracles

class Oracle(object):

    def evaluate(self,a):

        raise NotImplementedError()

    def update(self,a):

        raise NotImplementedError()

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
        #self.chosen_elements.append(chosen)
            #self.elements.remove(chosen.__getitem__(0))

        return chosen


class Information_utility(Oracle):
    '''Mutual information Utility'''
    def __init__(self,Pr_h,Ph,neuron_resp):
        self.Ph = Ph
        self.Pr_h = Pr_h
        self.neuron_resp = neuron_resp
        self.Phistory = [Ph]
    
    def evaluate(self,a):

        #
        #  nr - the number of responses, e.g., 80
        #  nh - the number of hypotheses, e.g., 9
        #
        #  Multiply a 2D array (nr x nh) by a 1D array (nh) to get a 1D array
        #  that is 'nr' long and contains the probability of observing
        #  any spike count value when stimulus 'a' is shown.
        #
        Pr = self.Pr_h[a,:,:].dot(self.Ph)

        #
        #  Compute the Mutual information:
        #
        #    I(r;h) = H(r) - H(r|h)
        #   
        #  which can be rewritten as  I(h;r) = H(h) - H(h|r), which is how
        #  Reza wrote it in his report.
        #
        #  The first term is H(r), where 'Pr' was computed by the previous line.
        #  The second term is H(r|h), which follows the definition of
        #  conditional entropy:
        #
        #    H(r|h) = SUM_h p(h) H(r|h)
        #
        #  The 'entropy(...)' part computes an array of entropies (of r) 
        #  for each hypothesis.  The ".dot" part multiplies this by the
        #  current prior for each hypothesis.
        #
        return entropy(Pr) - entropy(self.Pr_h[a,:,:]).dot(self.Ph)

    def update(self,a):

        #
        #  Get a (noisy) response from the neuron for stimulus 'a'
        #
        self.response = self.neuron_resp(a)

        #
        #  Apply Bayes' rule to update current probabilities of each hypoth.
        #
        self.Ph = self.Pr_h[a,self.response,:]*self.Ph/sum(self.Pr_h[a,self.response,:]*self.Ph)        
        self.Phistory.append(self.Ph)

        
class Uncertainty_utility(Oracle):
    '''Uncertainity Utility'''
    def __init__(self,Pr_h,Ph,neuron_resp):
        self.Ph = Ph
        self.Pr_h = Pr_h
        self.neuron_resp = neuron_resp
        self.Phistory = [Ph]

    def evaluate(self,a):
        Pr = self.Pr_h[a,:,:].dot(self.Ph)
        return entropy(Pr)
    
    def update(self,a):
        self.response = self.neuron_resp(a)
        self.Ph = self.Pr_h[a,self.response,:]*self.Ph/sum(self.Pr_h[a,self.response,:]*self.Ph)        
        self.Phistory.append(self.Ph)


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

def checkPearsonR(stim,responses,rot):
    expected = []
    for each in stim:
        expected.append(ideal[rot][each])

    #print(np.corrcoef(expected,list(responses.values()))[0][1])
    # find correlation coefficient between unknown cell's responses and ideal cell
    return np.corrcoef(expected,responses)[0][1]
    
def neuron_resp(a):
    #
    #  'a' is the index of the stimulus.
    #  'X' contains normalized responses in the range of 0..1
    #
    #  Return a random integer that is Poisson distributed with mean equal to
    #  50 times the mean responses of neuron '0' to stimulus 'a'
    #
    #  Added by Wyeth to control random seed
    
    return np.random.poisson(50*X[neuronnum-1][a])


###################
# INITIALIZE DATA #
###################

# Load normalized responses
X = np.load('./data/all_resp.npy')

# Prob of Hypotheses?  Clustering
probh = np.load('./data/probh50.npy')
ideal = np.load('./data/ideal50.npy')

# which neurons to test
neuronnum = int(input('Neuron: '))
seednum = int(input('Seed: '))
np.random.seed(seednum)

clusters = ['Red','Brown','Blue','Green','Purple','Gray']

#  ph - Array of current Pr[hypoth]
ph = np.array([.02]*16 + [0.2] + [.02]*24)

####################################
# Create Objects and Run Simulation #
####################################

f = open(str(neuronnum)+'-'+str(seednum)+'.csv','w')
f.write('Stimulus,Response,Ph')

# Mutual Information
print('\n'+'Mutual Information')
NC = Information_utility(probh,ph,neuron_resp)
GC = greedy(NC,list(range(362)),300)
count = [0]*6
stim = []
responses = []
rot = 0
for i in range(300):
    #seednum+=100

    f.write('\n')
    stimShown = GC.solve()

    stim.append(stimShown)
    f.write(str(stim[i])+',')
    responses.append(GC.oracle.response)
    f.write(str(responses[i])+',')
    for p in range(len(GC.oracle.Ph)):
        f.write(str(GC.oracle.Ph[p])+',')

    checkC = check(GC)
    if checkC != None:
        if checkPearsonR(stim,responses,rot) >= 0.6:
            print('Cluster: ' + clusters[checkC])
            print('Stimuli: ' + str(i))
            break
        else:
            count = [0]*6
    if i == 299 and checkC == None:
        print('Unclassified')
        break

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.prop_cycle'] = cycler('color',['r','#985114','b','g','m','0.40'])
NCP = []
for i in range(len(GC.oracle.Phistory)):
    NCP.append([sum(GC.oracle.Phistory[i][0:8]),
               sum(GC.oracle.Phistory[i][8:16]),
               GC.oracle.Phistory[i][16],
               sum(GC.oracle.Phistory[i][17:25]),
               sum(GC.oracle.Phistory[i][25:33]),
               sum(GC.oracle.Phistory[i][33:41])])

fig, ax = plt.subplots(1,1)
ax.plot(np.array(NCP))
ax.set_xlabel('number of stimuli sampled',fontsize = 16)
ax.set_ylabel('P(h)',fontsize = 16)

f.close()

plt.show()
plt.savefig(str(neuronnum)+'-'+str(seednum)+'.png')

'''
# Uncertainty
print('\n'+'Uncertainty')
NC_Dual = Uncertainty_utility(probh,ph,neuron_resp)
GC_Dual = greedy(NC_Dual,list(range(362)),300)
count = [0]*6
responses = {}
rot = 0
for i in range(300):
    stimShown = GC_Dual.solve()

    responses[stimShown] = GC_Dual.oracle.response
    checkC = check(GC_Dual)
    if checkC != None:
        if checkPearsonR(responses,rot) >= 0.6:
            print('Cluster: ' + clusters[checkC])
            print('Stimuli: ' + str(i))
            break
    if i == 299 and checkC == None:
        print('Unclassified')
        break
'''
