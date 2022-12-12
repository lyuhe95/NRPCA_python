import numpy as np
from random import choices
import math
from scipy.spatial.distance import cdist
from collections import Counter

def get_swiss(N,sparse_noise_level,gaussian_noise_level=0):
    # Construct swiss roll of dim 20
    P = 20
    t = 4*np.pi*np.random.rand(N,1)
    t = np.array(sorted(t))
    x = (t+1)*np.cos(t)
    y = (t+1)*np.sin(t)
    z = 8*math.pi*np.random.rand(N,1)

    t_max = np.max(t)**2+1
    c = 256/t_max*(t**2+1)
    X = np.concatenate((x,y,z),axis=1)
    for d in range(4,21):
        tmp = t*np.sin(d*t/21).reshape(N,1)
        X = np.concatenate((X,tmp),axis=1)
    # Add sparse noise
    num_outliers = math.ceil(N*P*0.015)
    outlier_loc = choices(list(np.arange(N)),k=num_outliers)
    outlier_dim = choices(list(np.arange(2))+list(np.arange(3,P+1)),k=num_outliers)
    for r,c in zip(outlier_loc,outlier_dim):
        X[r][c] += np.random.normal(sparse_noise_level,0.03)*int(random.random()<0.5)
    # Add gaussian noise
    X += gaussian_noise_level*np.random.randn(N,20)
    return X

def distance_matrix(X,K):
    D = sp.spatial.distance.cdist(X,X)
    N = D.shape[0]
    num_copies = Counter()
    D1 = np.zeros((N,K))
    for i in range(N):
        indices = np.argsort(D[i])[:K+1]
        for j in indices:
            if j != i:
                D[i,j] = 1
                num_copies[j] += 1
    A = D*D1
    return A,num_copies
        
        