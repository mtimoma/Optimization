from __future__ import division
import numpy as np
import sys


minCorr = 0.6
####################
# Creating Objects #
####################

class Oracle(object):

    def evaluate(self,a):
        raise NotImplementedError()

    def update(self,a):
        raise NotImplementedError()

def argmax(oracle,all_elements):

    max_imp = -999999999

    for a in all_elements:

        # prevent it from choosing stimulus chosen in last 2 stim
        if a in stim[-1:-3:-1]:
            continue
        # don't show a stimulus that has already been shown 10 times
        if stim.count(a) > 10:
            continue

        imp = oracle.evaluate(a)

        if imp > max_imp:
            max_imp = imp
            b = a

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

'''Mutual information Utility'''
class Information_utility(Oracle):

    def __init__(self,Pr_h,Ph,neuron_resp):
        self.Ph = Ph
        self.Pr_h = probh[meanresp[self.Ph.tolist().index(max(self.Ph))]-1]
        self.neuron_resp = neuron_resp
        self.Phistory = [Ph]
    
    def evaluate(self,a):
        # use probability vector with mean for hypothesis with greatest probability
        self.Pr_h = probh[meanresp[self.Ph.tolist().index(max(self.Ph))]-1]
        #  get a 1D array of Poisson probability
        Pr = self.Pr_h[a,:,:].dot(self.Ph)
        #  Compute the Mutual information:
        return entropy(Pr) - entropy(self.Pr_h[a,:,:]).dot(self.Ph)

    def update(self,a):

        #self.Ph = self.Ph.tolist()
        self.Pr_h = probh[meanresp[self.Ph.tolist().index(max(self.Ph))]-1]
        #  Get a (noisy) response from the neuron for stimulus 'a'
        self.response = self.neuron_resp(a)

        #  Apply Bayes' rule to update current probabilities of each hypoth.
        self.Ph = self.Pr_h[a,self.response,:]*self.Ph/sum(self.Pr_h[a,self.response,:]*self.Ph)        
        self.Phistory.append(self.Ph)

# entropy equation used for Mutual Information
# used because scipy may not run on machines
def entropy(P):
    np.seterr(all='ignore')
    return -sum(P * np.log(P))


##################################
# Criteria for Returning Cluster #
##################################

# check if hypothesis has probability greater than 0.999
#                                     greater than 10x
def check(GC):
    global count
    global rot

    # sum all rotations of each cluster
    prob_red = sum(GC.oracle.Ph[0:8])
    prob_brown = sum(GC.oracle.Ph[8:16])
    prob_blue = GC.oracle.Ph[16]
    prob_green = sum(GC.oracle.Ph[17:25])
    prob_purple = sum(GC.oracle.Ph[25:33])
    prob_gray = sum(GC.oracle.Ph[33:41])
    prob_uncl = GC.oracle.Ph[41]

    # check if any hypothesis has probability greater than 0.999
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
    elif prob_uncl > 0.999:
        count[6] += 1
    else:
        # divide count by 2 if no probability is greater than 0.999
        count = (np.array(count)/2).astype(int).tolist()

    # if hypothesis had probability greater than 0.999 for last 10 times
    # return that hypothesis
    if 10 in count:
        return count.index(10)

# check r-value between responses and an ideal cell of same cluster
def checkPearsonR(stim,responses,rot,clus):
    global ccount

    allcorr = []
    # normalize responses and multiply by 25
    # this helps to make r-value more accurate
    responses=np.asarray(responses)*25/meanresp[rot]
    # for every core member of the group
    for core in range(len(ideal[24][rot])):
        # create vector of expected responses based on what set of stimuli were shown
        expected = []
        for each in stim:
            expected.append(ideal[24][rot][core][each])

        # find correlation coefficient between unknown cell's responses and ideal cell
        allcorr.append(np.corrcoef(expected,responses)[0][1])
    # return the largest adjusted correlation coefficient
    ccount[clus] += 1
    return max(allcorr)*(1+(1-max(allcorr)**2)/(2*len(stim)))


#######################################
# Getting Neuron Response to Stimulus #
#######################################

# Load normalized responses
X = np.load('./data/all_resp.npy')
# Load max Firing Responses
maxF = []
with open("RawData/MaxFiringRate/max_resp.txt") as maxFire:
    n = maxFire.read().splitlines()
for i in range(len(n)):
    maxF.append(float(n[i]))

def neuron_resp(a):
    #  Return a noise response to stimulus 'a'
    resp = np.random.poisson(maxF[neuronnum]*X[neuronnum][a])
    if resp < 80:
        return resp
    else:
        return 79


###################
# INITIALIZE DATA #
###################

# load probability vectors for all sizes
probh = np.load('./data/probh.npy')
ideal = np.load('./data/ideal.npy')
norms = np.load('./data/maxr.npy')
# which neuron to test
cells = int(input('# of Neurons: '))
trials = int(input('# of Trials: '))

clusters = ['Red','Brown','Blue','Green','Purple','Gray']

#  ph - Array of current Pr[hypoth]
ph = np.array([.01]*8 + [.004]*8 + [.14] + [.002]*8 + [.003]*8 + [.002]*8 + [.692])

# open file
f = open('MutualAll.csv','w')
for i in range(trials):
    f.write('Cluster,Stimuli,Size,')

# show progress of script
sys.stdout.write('0/' + str(cells))
sys.stdout.flush()


#################################
# Find Means for All Hypotheses #
#################################

# find mean response for each hypothesis by dividing response by normalized
def findMean(stim,resp):
    mr = np.array([0]*42)
    for s in range(len(stim)):
        for i in range(42):
            if norms[i][stim[s]] == 0:
                mr[i] = mr[i] + mr[i]/len(stim)
            else:
                mr[i] += resp[s] / norms[i][stim[s]]
    mr = mr / len(stim)

    # if any mean is greater than 25, change it to 25
    for resp in range(len(mr)):
        if mr[resp] > 24:
            mr[resp] = 24
        else:
            mr[resp] -= 1

    return mr

for n in range(cells):
    f.write('\n')
    neuronnum = n
    for s in range(trials):
        np.random.seed(1000+s*123)

        # choose 5 random stimulus
        initStim = np.random.randint(362, size=5)
        initResponse = []
        for stim in initStim:
            initResponse.append(neuron_resp(stim))

        meanrespUR = findMean(initStim,initResponse)
        meanresp = np.ceil(meanrespUR).astype(int)
        for mean in meanresp:
            if mean == 0:
                mean += 1


        #####################################
        # Create Objects and Run Simulation #
        #####################################

        # Mutual Information
        NC = Information_utility(probh,ph,neuron_resp)
        GC = greedy(NC,list(range(362)),300)

        # how many times the hypothesis reaches 0.999 probability
        count = [0]*7
        ccount = [0]*6

        # stores stimuli, responses, and rotation to calculate r-value
        stim = []
        responses = []
        rot = 0

        for i in range(300):
            stimShown = GC.solve()

            # create lists of stimuli and responses for later use
            stim.append(stimShown)
            responses.append(GC.oracle.response)

            # check if any hypothesis has count above 10 
            checkC = check(GC)
            if checkC != None:
                if checkC == 6:
                    f.write('Unclassified,')
                    f.write(str(i+5) + ',')
                    f.write(str(meanresp[41]) + ',')
                    break
                # if r value over minimum correlation set, return cluster chosen
                if checkPearsonR(stim,responses,rot,checkC) >= minCorr:
                    f.write(clusters[checkC] + ',')
                    f.write(str(i+5) + ',')
                    f.write(str(meanresp[rot]) + ',')
                    break
                else:
                    count = [0]*6
                    checkC = None

            # return 'Unclassified' if chose same color 3 times but r-value too low
            if 3 in ccount:
                f.write('Unclassified,')
                f.write(str(i+5) + ',')
                f.write(str(meanresp[41]) + ',')
                break

            # return 'Unclassified' if reach max stimuli
            if i == 299 and checkC == None:
                f.write('Unclassified,')
                f.write(str(i+5) + ',')
                f.write(str(meanresp[41]) + ',')
                break

            # Recalculate means
            meanrespUR = ((i+5)*meanrespUR + findMean([stim[i]],[responses[i]])) / (i+6)
            meanresp = np.ceil(meanrespUR).astype(int)
            for m in range(len(meanresp)):
                if meanresp[m] < 1:
                    meanresp[m] += 1

            # Recalculate Ph after 10 stimuli
            if i%10 == 0 and i > 0:
                Ph = np.array([.01]*8 + [.004]*8 + [.14] + [.002]*8 + [.003]*8 + [.002]*8 + [.692])
                for a in range(len(stim)):
                    Pr_h = probh[meanresp[Ph.tolist().index(max(Ph))]-1]
                    Ph = Pr_h[stim[a],responses[a],:]*Ph/sum(Pr_h[stim[a],responses[a],:]*Ph)
                GC.oracle.Ph = Ph

    # update progress number            
    sys.stdout.write('\r')
    sys.stdout.write(str(n+1) + '/' + str(cells))
    sys.stdout.flush()

f.close()