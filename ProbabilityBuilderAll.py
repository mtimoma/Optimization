import numpy as np
from scipy.stats import poisson
from scipy.stats import pearsonr
import sys
#from tempfile import TemporaryFile


##########################
# OPENING RAW DATA FILES #
##########################

# Opens NormResponse files for first 215 neurons
resp = []
for i in range(215):
    respf = []
    # store responses of each cell in python array
    with open("RawData/NormResponse/%s" % (i + 1)) as norm:
        respf = norm.read().splitlines()
        resp.append(respf)
# change array of all responses into numpy float array
resph = np.asarray(resp).astype(np.float)

# Open Clusters.txt into array
cluster = []
with open("RawData/Clusters/Training_Clusters.txt") as clusters:
    c = clusters.read().splitlines()
for each in c:
    cloos = []
    # splits the text file into an array
    neach = each.split(',')
    # removes the cluster name from array
    del neach[0]

    # separate neuron number from r-value
    for every in neach:
        if '`' in every:
            neweach = every.split('`')
        else:
            neweach = [every,1]
        neweach = list(map(int, neweach))
        cloos.append(neweach)

    # returns array with neuron # in cluster and r-value for each
    cluster.append(cloos)


###########################
# FIND MEAN NORMALIZATION #
###########################

for nnum in range(len(resph)):
    resph[nnum] = resph[nnum] / np.mean(resph[nnum])


##################
# ALIGN ROTATION #
##################

# Initialize rotation map
glob_nrot = np.zeros(51).tolist()
glob_nrot[0] = 1
glob_nrot[1] = 1
glob_nrot[2] = 8
glob_nrot[3] = 4
glob_nrot[4] = 4
glob_nrot[5] = 8
glob_nrot[6] = 4
for i in range(7, 31):
    glob_nrot[i] = 8
glob_nrot[31] = 2
glob_nrot[32] = 8
glob_nrot[33] = 4
glob_nrot[34] = 8
glob_nrot[35] = 8
glob_nrot[36] = 2
for i in range(37, 44):
    glob_nrot[i] = 8
glob_nrot[44] = 4
for i in range(45, 51):
    glob_nrot[i] = 8

glob_shi0 = [0]
for i in range(1, 51):
    glob_shi0.append(glob_shi0[i - 1] + glob_nrot[i - 1])

# Rotate response and compare it
for color in cluster:
    # for every group other than blue
    if color != cluster[2]:
        # set first neuron in group as template
        resp_main = resph[color[0][0] - 1].tolist()

        for neuron in color:

            checkval = 0
            # rotate a total of 8 times
            for rot in range(8):
                for i in range(51):
                    nrot = glob_nrot[i]
                    k = glob_shi0[i]

                    if nrot > 1:
                        tr = resph[neuron[0] - 1][k]
                        for j in range(1, nrot):
                            resph[neuron[0] - 1][k + j - 1] = resph[neuron[0] - 1][k + j]
                        resph[neuron[0] - 1][k + nrot - 1] = tr

            # Compare arrays
                corr = pearsonr(resph[neuron[0] - 1], resp_main)[0]
            # If pearson r value is highest, use that rotation
                if corr > checkval:
                    newresph = resph[neuron[0] - 1].tolist()
                    checkval = corr
            resph[neuron[0] - 1] = newresph


###########################
# FINDING MEAN NORMALIZED #
###########################
Pr_h = []

# RED and BROWN
for clusnum in range(2):
    clusclass = np.zeros(362)
    for neuron in cluster[clusnum]:
        # sum product of normalized responses and r-value
        clusclass = clusclass + resph[int(neuron[0]) - 1]*int(neuron[1])
    # average normalized responses by dividing by sum of r-values
    clusclass = clusclass / np.sum(cluster[clusnum],axis=0)[1]

    # add first rotation to complete probability vector
    Pr_h.append(clusclass.tolist())

    # add rotations 2-8 to complete probability vector
    for rots in range(7):
        for i in range(51):
            nrot = glob_nrot[i]
            k = glob_shi0[i]
            if nrot > 1:
                tr = clusclass[k]
                for j in range(1, nrot):
                    clusclass[k + j - 1] = clusclass[k + j]
                clusclass[k + nrot - 1] = tr
        Pr_h.append(clusclass.tolist())

# BLUE
blue = np.zeros(362)
for neuron in cluster[2]:
    # add mean response together for each stimulus
    blue = blue + resph[int(neuron[0]) - 1]*int(neuron[1])
# average normalized mean response from blue cluster neurons per stimulus
blue = blue / np.sum(cluster[2], axis = 0)[1]
Pr_h.append(blue)

#GREEN and PURPLE and GRAY
for clusnum in range(3,6):
    clusclass = np.zeros(362)
    for neuron in cluster[clusnum]:
        # sum product of normalized responses and r-value
        clusclass = clusclass + resph[int(neuron[0]) - 1]*int(neuron[1])
    # average normalized responses by dividing by sum of r-values
    clusclass = clusclass / np.sum(cluster[clusnum],axis=0)[1]

    # add first rotation to complete probability vector
    Pr_h.append(clusclass.tolist())

    # add rotations 2-8 to complete probability vector
    for rots in range(7):
        for i in range(51):
            nrot = glob_nrot[i]
            k = glob_shi0[i]
            if nrot > 1:
                tr = clusclass[k]
                for j in range(1, nrot):
                    clusclass[k + j - 1] = clusclass[k + j]
                clusclass[k + nrot - 1] = tr
        Pr_h.append(clusclass.tolist())

# Unclassified
uncl = []
for size in range(1,26):
    unc = []
    for i in range(80):
        if i <= np.pi*size:
            unc.append(np.cos([i/size])[0]+1.01)
        else:
            unc.append(unc[len(unc)-1])
    uncl.append(unc)


##############################
# CREATING PROBABILITY ARRAY #
##############################
print('Creating Poisson Distributions...')
sys.stdout.write('0/25')
sys.stdout.flush()
# creates 25x362x80x41 array to cover all clusters
probh = []
for size in range(1,26):
    probh_m = []
    for i in range(len(Pr_h[0])):
        pro = []
        for j in range(80):
            pr = []
            for k in range(len(Pr_h)):
                # Poisson probabilities for responses 0-80 given mean
                pr.append(poisson.pmf(j, Pr_h[k][i] * size, loc=0))
            pr.append(uncl[size-1][j]/np.sum(uncl[size-1]))
            pro.append(pr)
        probh_m.append(pro)
    probh.append(probh_m)
    sys.stdout.write('\r')
    sys.stdout.write(str(size) + '/25')
    sys.stdout.flush()

np.save("DATA/probh", probh)

print('\nCreating Normalized Vectors...')
sys.stdout.write('0/' + str(len(Pr_h)))
sys.stdout.flush()
#used to initially find means for all hypotheses
maxr = []
for i in range(len(Pr_h)):
    max_r = []
    for j in range(len(Pr_h[0])):
        max_r.append(Pr_h[i][j])
    maxr.append(max_r)
    sys.stdout.write('\r')
    sys.stdout.write(str(i+1) + '/' + str(len(Pr_h)))
    sys.stdout.flush()
maxr.append([1]*len(Pr_h[0]))

np.save("DATA/maxr",maxr)


#########################################
# Create Ideal Cells for R Calculations #
#########################################

# Open cores.txt into array
print('\nCreating Core Cells...')
sys.stdout.write('0/25')
sys.stdout.flush()
cores = []
with open("RawData/Clusters/Training_Cores.txt") as core:
    co = core.read().splitlines()

for each in co:
    # splits the text file into an array
    ceach = each.split(',')
    # removes the cluster name from array
    del ceach[0]
    cores.append(list(map(int, ceach)))

# create ideal vectors for all cores and rotations
ideal = []
for size in range(1,26):
    idea = []
    for hyp in range(6):
        # no need for rotation if cluster is blue
        if hyp == 2:
            ide = []
            for h in cores[hyp]:
                ide.append(resph[h-1] * size)
            idea.append(ide)

        # include 8 rotations for all clusters that are not blue
        else:
            ide = []
            for h in cores[hyp]:
                ide.append(resph[h-1] * size)
            idea.append(ide)
            for rot in range(7):
                ide = [] 
                for h in cores[hyp]:
                    for i in range(51):
                        nrot = glob_nrot[i]
                        k = glob_shi0[i]

                        if nrot > 1:
                            tr = resph[h - 1][k]
                            for j in range(1, nrot):
                                resph[h - 1][k + j - 1] = resph[h - 1][k + j]
                            resph[h - 1][k + nrot - 1] = tr                  
                    ide.append(resph[h-1] * size)
                idea.append(ide)
                
    ideal.append(idea)
    sys.stdout.write('\r')
    sys.stdout.write(str(size) + '/25')
    sys.stdout.flush()

np.save("DATA/ideal", ideal)
