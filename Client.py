import socket
import numpy as np

def neuron_resp(a):
    #
    #  'a' is the index of the stimulus.
    #  'X' contains normalized responses in the range of 0..1
    #
    #  Return a random integer that is Poisson distributed with mean equal to
    #  50 times the mean responses of neuron '0' to stimulus 'a'
    #
    #  Added by Wyeth to control random seed
    np.random.seed(seednum)
    return np.random.poisson(50*X[neuronnum-1][a])

# Load normalized responses
X = np.load('./data/all_resp.npy')

r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

port = 8888
remote_ip = '192.168.0.105'

r.connect((remote_ip , port))

HOST = ''
PORT = 8000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()

neuronnum = int(r.recv(1024).decode())
seednum = int(r.recv(1024).decode())

while True:
	stim = r.recv(1024).decode()
	if stim == 'Done':
		break
	print(stim)
	response = neuron_resp(int(stim))
	conn.sendall(str(response).encode())

r.close()