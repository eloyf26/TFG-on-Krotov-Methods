import qutip
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov

def generate_arrays(d1 = [], d2 = []):
    # Create array with d1 on the diagonal
    arr1 = np.diag(d1)
    
    # Create array with d2 on the upper and lower diagonals
    arr2 =   np.diag(d2,1) + np.diag(d2,-1)# lower diagonal

    return arr1, arr2

arr1,arr2 = generate_arrays(d1,d2); 
D
diag = []
for i in range(Dim):
    diag.append((i+1)**2/4)

d1 = diag
d2 = np.ones((1,Dim-1)) * 2
H = hamiltonian(N=Dim,d1=d1,d2=d2)