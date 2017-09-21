import numpy as np
import socket


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = 'Self'
port = 8888

remote_ip = '192.168.0.104'

s.connect((remote_ip , port))
print ('Socket Connected to ' + host + ' on ip ' + remote_ip)

#######################################
# Getting Neuron Response to Stimulus #
#######################################
def neuron_resp(a):
    #  Return a noise response to stimulus 'a'
    resp = np.random.poisson(maxF[neuronnum-1]*X[neuronnum-1][a])
    if resp < 80:
        return resp
    else:
        return 79

# Load normalized responses
X = np.load('./data/all_resp.npy')
# Load max Firing Responses
maxF = []
with open("RawData/MaxFiringRate/max_resp.txt") as maxFire:
    n = maxFire.read().splitlines()
for i in range(len(n)):
    maxF.append(float(n[i]))

neuronnum = int(s.recv(1024).decode('utf-8'))
seednum = int(s.recv(1024).decode('utf-8'))
np.random.seed(seednum)

f = open('Responses','w')
while True:
    stim = int(s.recv(1024).decode('utf-8'))
    print(stim)
    if stim == 4000:
        break
    f.write('\n'+str(neuron_resp(stim)))
    f.flush()

f.close()
s.close()