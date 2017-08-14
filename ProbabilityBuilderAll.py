import numpy as np
from scipy.stats import poisson
from scipy.stats import pearsonr


##########################
# OPENING RAW DATA FILES #
##########################

#
# Opens NormResponse files for first 172 neurons
resp = []
for i in range(215):
    respf = []
    # store responses of each cell in python array
    with open("RawData/NormResponse/%s" % (i + 1)) as norm:
        respf = norm.read().splitlines()
        resp.append(respf)
# change array of all responses into numpy float array
resph = np.asarray(resp).astype(np.float)

#
# Open Clusters.txt into array
cluster = []
with open("RawData/Clusters/Clusters.txt") as clusters:
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

##############################
# CREATING PROBABILITY ARRAY #
##############################
probh = []

# creates 25x362x80x41 array to cover all clusters
for size in range(1,26):
    probh_m = []
    for i in range(len(Pr_h[0])):
        pro = []
        for j in range(80):
            pr = []
            for k in range(len(Pr_h)):
                # Poisson probabilities for responses 0-120 given mean
                pr.append(poisson.pmf(j, Pr_h[k][i] * size, loc=0))
            pro.append(pr)
        probh_m.append(pro)
    probh.append(probh_m)

np.save("DATA/probh", probh)

#creates 'ideal' cells for each group
#array of size 25x41x362
ideal = []

for size in range(1,26):
    idem = []
    for i in range(len(Pr_h)):
        ide = []
        for j in range(len(Pr_h[0])):
            ide.append(Pr_h[i][j] * size)
        idem.append(ide)
    ideal.append(idem)

np.save("DATA/ideal", ideal)

maxr = []
    for i in range(len(Pr_h)):
        max_r = []
        for j in range(len(Pr_h[0])):
            max.append(Pr_h[i][j])
        maxr.append(max_r)

np.save("DATA/maxr",maxr)