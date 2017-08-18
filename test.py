import sys

print('Creating Poisson Distributions...')
sys.stdout.write('0/26')
# creates 25x362x80x41 array to cover all clusters
probh = []
for size in range(1,26):
    probh_m = []
    for i in range(100):
        probh_m.append(i*size)
    probh.append(probh_m)
    #sys.stdout.write('\r')
    #sys.stdout.write(str(size+1) + '/26')
    #sys.stdout.flush()

print('\nCreating Normalized Vectors...')
#used to initially find means for all hypotheses
maxr = []
for i in range(41):
    max_r = []
    for j in range(10):
        max_r.append(j*i)
    maxr.append(max_r)
    sys.stdout.write('\r')
    sys.stdout.write(str(i) + '/41')
    sys.stdout.flush()