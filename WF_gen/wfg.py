import numpy as np
import time

# Performs tensorproduct between two vectors c=a(x)b
def TensProd(a, b):
	c=np.zeros(len(a)*len(b), dtype=complex)
	for ii in range(len(b)):
		c[ii*len(a):(ii*len(a)+len(a))]=b[ii]*a[:]
	return c

# Generates a random normalized wavefunction
def RandomWF(DD):
	psi=np.random.normal(size=DD)+np.random.normal(size=DD)*1j
	psi=psi/np.linalg.norm(psi)
	return psi


DD=2 # Hilbert space size
Npart=10	# Num of subparts
Nsep=4000 # Num of sep wf to campionate
Nnsep=4000 # Num of entangled wf to campionate

# Separable wf generation
t0=time.time()	

a=RandomWF(DD)
for jj in range(Npart-1):
	b=RandomWF(DD)
	a=TensProd(a, b)
	
sep=a.reshape(1,-1)
for ii in range(Nsep-1):
	a=RandomWF(DD)
	for jj in range(Npart-1):
		b=RandomWF(DD)
		a=TensProd(a, b)
	sep=np.concatenate((sep,a.reshape(1,-1)), axis=0)	
print('Generated an array of shape: {}').format(sep.shape)	

sepRE=np.real(sep).reshape(Nsep, DD**Npart, 1)
sepIM=np.imag(sep).reshape(Nsep, DD**Npart, 1)
np.save('sep.npy', np.concatenate((sepRE,sepIM), axis=2))

t1=time.time()
print('Time to generate separable wf: '+str(t1-t0))

# Entangled wf generation
t0=time.time()	

nsep=RandomWF(DD**Npart).reshape(1,-1)
for ii in range(Nnsep-1):
	a=RandomWF(DD**Npart).reshape(1,-1)
	nsep=np.concatenate((nsep,a), axis=0)
print('Generated an array of shape: {}').format(nsep.shape)
	
nsepRE=np.real(nsep).reshape(Nnsep, DD**Npart, 1)
nsepIM=np.imag(nsep).reshape(Nnsep, DD**Npart, 1)
np.save('nsep.npy', np.concatenate((nsepRE,nsepIM), axis=2))

t1=time.time()
print('Time to generate entangled wf: '+str(t1-t0))


