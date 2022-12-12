import numpy as np
from random import choices
import math
from utils import *

class NRPCA:
    
    def __init__(self,X,K,num_runs,niter):
        self.noisy_X = X
        self.K = K
        self.N = X.shape[0]
        self.num_runs = num_runs
        self.niter = niter
        
    def run(self):
        L_temp = self.noisy_X
        lambdas = [1]*self.N
        for i in range(self.num_runs):
            print(f'Start round {i} of NRPCA ...')
            A,patch,num_copies = self.distance_matrix(L_temp)
            L_temp = self.solver(patch,num_copies)
            L.concatenate(L_temp,axis=1)
        return L
            
    def distance_matrix(self,X):
        D = sp.spatial.distance.cdist(X,X)
        num_copies = Counter()
        patch = {i:[i] for i in range(self.N)}
        D1 = np.zeros((self.N,self.K))
        for i in range(self.N):
            indices = np.argsort(D[i])[:K+1]
            for j in indices:
                if j != i:
                    D[i,j] = 1
                    num_copies[j] += 1
                    patch[i].append(j)
        A = D*D1
        return A,patch,num_copies
    
    def solver(self,patch,num_copies):
        print('Start solving minimization problem...

