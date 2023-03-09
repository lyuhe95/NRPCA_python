import numpy as np
import random
from random import choices
from collections import defaultdict,Counter
import math
import scipy as sp
from scipy.spatial.distance import cdist
from utils import *
import argparse
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import ipdb
import time

class NRPCA:
    
    def __init__(self,X,K,num_runs,niter,gaussian=False):
        self.noisy_X = X
        self.K = K
        self.N = X.shape[0]
        self.P = X.shape[1]
        self.num_runs = num_runs
        self.niter = niter
        self.gaussian = gaussian
        self.C = np.eye(self.K+1) - 1/(self.K+1)*np.ones((self.K+1,self.K+1))
        
    def run(self):
        L = defaultdict()
        L_temp = self.noisy_X
        for i in range(self.num_runs):
            print(f'Start round {i} of NRPCA ...')
            A,patch,num_copies = self.distance_matrix(L_temp)
            L_temp = self.solver(patch,num_copies)
            L[i] = L_temp
        if self.gaussian:
            X_hat = self.clean_L(self.noisy_X-L_temp,patch,num_copies)
        else:
            X_hat = L_temp
        return L,X_hat
    
    def solver(self,patch,num_copies):
        print('Start solving minimization problem...')
        #ipdb.set_trace()
        S = np.zeros_like(self.noisy_X)
        t = 0.005
        theta = 1
        S_old = S
        mu = np.array([num_copies[i]/np.sqrt(max(self.K,self.P)) for i in range(self.N)]).reshape(-1,1)
        for i in range(self.niter):
            print(f'Iter {i}')
            S_new = self.g_prox(S - t * self.f_grad(S,patch),t,mu)
            theta_new = 0.5 * (1 + np.sqrt(1 + 4 * theta**2))
            S = S_new + (theta - 1.)/theta_new * (S_new - S_old)
            S_old = S_new
            theta = theta_new
            obj_val = self.f_val(S_new,patch) + self.g_val(S_new,mu)
            print(f'Objective value: {obj_val}')         
        L = self.noisy_X - S_new
        return L
    
    def distance_matrix(self,X):
        D = cdist(X,X)
        num_copies = defaultdict(lambda:1)
        patch = {i:[i] for i in range(self.N)}
        D1 = np.zeros_like(D)
        for i in range(self.N):
            indices = np.argsort(D[i])[:self.K+1]
            for j in indices:
                if j != i:
                    D1[i,j] = 1
                    num_copies[j] += 1
                    patch[i].append(j)
        A = D*D1
        return A,patch,num_copies
        
    def f_grad(self,S,patch):
        y = np.zeros_like(self.noisy_X)
        for i in range(self.N):
            neighbors = patch[i]
            S_i = self.C@S[neighbors]
            X_i = self.C@self.noisy_X[neighbors]
            [U,Sigma_i,Vh] = np.linalg.svd(X_i-S_i,full_matrices=False)
            Sigma_i = [1]*len(Sigma_i)
            L_i = U@np.diag(Sigma_i)@Vh
            temp = -self.C@L_i 
            y[neighbors] += temp
        return y
    
    def f_val(self,S,patch):
        y = 0
        for i in range(self.N):
            neighbors = patch[i]
            S_i = self.C@S[neighbors]
            X_i = self.C@self.noisy_X[neighbors]
            Sigma_i = np.linalg.svd(X_i-S_i,full_matrices=False,compute_uv=False)
            tmp = sum(Sigma_i)
            y += tmp
        return y
    
    def g_prox(self,S,t,mu):
        y = np.maximum(0,S-t*mu*np.ones((1,self.P))) - np.maximum(0,-S-t*mu*np.ones((1,self.P)))
        return y
    
    def g_val(self,S,mu):
        y = 0
        for i in range(self.N):
            y += mu[i] * sum(abs(S[i]))
        return y

    def optimal_thresh(self):
        beta = min(self.P/self.K, self.K/self.P);
        return np.sqrt(2*(beta+1)+8*beta/((beta+1)+np.sqrt(beta**22+14*beta+1)))
    
    def clean_L(self,S,patch,num_copies):
        X_hat = np.zeros_like(self.noisy_X)
        thresh = self.optimal_thresh()
        for i in range(self.N):
            neighbors = patch[i]
            S_i = S[neighbors]
            X_i = self.noisy_X[neighbors]
            [U,Sigma_i,Vh] = np.linalg.svd(self.C@(X_i-S_i))
            Sigma_i[Sigma_i < thresh] = 0
            L_i = np.dot(U,np.dot(sp.linalg.diagsvd(Sigma_i,*X_i.shape),Vh)) + (np.eye(self.K+1)-self.C)@(X_i-S_i)
            X_hat[neighbors] += L_i
        for i in range(self.N):
            X_hat[i] = [x/num_copies[i] for x in X_hat[i]]
        return X_hat
            
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run NRPCA.")
    parser.add_argument('-data', type=str, default='Swiss', help='Swiss or MNIST')
    parser.add_argument('-N', type=int, default=2000, help='Number of datapoints')
    parser.add_argument('-K', type=int, default=15, help='Number of neighbors')
    parser.add_argument('-Sparse_noise_level', type=float, default=6., help='Sparse noise level, only for swiss')
    parser.add_argument('-Gaussian_noise_level', type=float, default=0.5,help='Gaussian noise level, only for swiss')
    parser.add_argument('-num_runs', type=int, default=2,help='Number of updating neighbors')
    parser.add_argument('-niter', type=int, default=100,help='Number of iters')
    return parser.parse_args()

def main():
    args = parse_args()
    N,K,sparse_noise_level,gaussian_noise_level = args.N,args.K,args.Sparse_noise_level,args.Gaussian_noise_level
    num_runs,niter = args.num_runs,args.niter
    if args.data.startswith('Swiss'):
        X,clean_X,color = get_swiss(N,sparse_noise_level=sparse_noise_level,gaussian_noise_level=gaussian_noise_level)
        gaussian = gaussian_noise_level > 0
    elif args.data.startswith('MNIST'):
        X,_ = get_mnist49(N)
        clean_X,color,gaussian = None,None,False
    
    method = NRPCA(X,K,num_runs,niter,gaussian=gaussian)
    L,X_hat = method.run()
    return X,L,X_hat,clean_X,color

if __name__ == '__main__':
    X,L,X_hat,clean_X,color = main()
            
            
            

