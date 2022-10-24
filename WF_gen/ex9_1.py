import subprocess as sp
import numpy as np
import os
import time  

t0=time.time()
# List of dimensions to try
Nlist=[10]
# Number of lambda in range Start-Stop
Nlambda=600
Start=-3.0
Stop=3.0
# Number of eigenvalues to save
KMAX=1


# Run the Fortran code with the choosen parameters
count=0
for NN in Nlist:
	t1=time.time()
	for ll in np.linspace(Start, Stop, Nlambda):
		of1=open('parameters.txt', 'w')
		of1.write(str(NN)+'	'+str(ll)+'	'+str(KMAX))
		of1.close()
		sp.call('./wfg')
		count += 1
		if(np.mod(count,20)==0): print(str(count)+'th lambda done')
	t2=time.time() 
	print('Mean execution time for lambda='+str((t2-t1)/Nlambda)+' s ')

    

