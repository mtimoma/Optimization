import numpy as np

# Opens NormResponse files for first 215 neurons
resp = []
for i in range(172):
    respf = []
    # store responses of each cell in python array
    with open("RawData/NormResponse/%s" % (i + 1)) as norm:
        respf = norm.read().splitlines()
        resp.append(respf)
# change array of all responses into numpy float array
resph = np.asarray(resp).astype(np.float)

# Load normalized responses
X = np.load('./data/all_resp.npy')
# Load max Firing Responses
maxF = []
with open("RawData/MaxFiringRate/max_resp.txt") as maxFire:
    n = maxFire.read().splitlines()
for i in range(len(n)):
    maxF.append(float(n[i]))

with open('HistogramS.csv','w') as f:
	for i in range(172):
		if maxF[i] * np.mean(resph[i]) <= 5:
			for x in X[i]:
				f.write(str(x/np.mean(resph[i])) + '\n')

with open('HistogramM.csv','w') as f:
	for i in range(172):
		if 5 < maxF[i] * np.mean(resph[i]) <= 10:
			for x in X[i]:
				f.write(str(x/np.mean(resph[i])) + '\n')

with open('HistogramL.csv','w') as f:
	for i in range(172):
		if 10 < maxF[i] * np.mean(resph[i]) <= 20:
			for x in X[i]:
				f.write(str(x/np.mean(resph[i])) + '\n')

with open('HistogramXL.csv','w') as f:
	for i in range(172):
		if 20 < maxF[i] * np.mean(resph[i]):
			for x in X[i]:
				f.write(str(x/np.mean(resph[i])) + '\n')
