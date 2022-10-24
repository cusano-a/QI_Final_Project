from scipy.io import FortranFile
import numpy as np
import time 

t0=time.time()
mode='eval'
N=600 # number of wavefunctions in the set
if(mode=='evec'):
	f = FortranFile('evec.bin', 'r')
	x=np.array(f.read_reals(dtype=np.complex64)).reshape(1,1025)
	for i in range(N-1):
		y=np.array(f.read_reals(dtype=np.complex64)).reshape(1,1025)
		x=np.concatenate((x,y), axis=0)
	f.close()
	outfile='evec.npy'
elif(mode=='eval'):
	f = FortranFile('eval.bin', 'r')
	x=np.array(f.read_reals(dtype=np.float32)).reshape(1,2)
	for i in range(N-1):
		y=np.array(f.read_reals(dtype=np.float32)).reshape(1,2)
		x=np.concatenate((x,y), axis=0)
	f.close()
	outfile='eval.npy'


np.save(outfile, np.real(x))

t1=time.time()-t0
print('Time to acquire the data: '+str(t1))
print('Shape of the array acquired: '+str(np.shape(x)))
