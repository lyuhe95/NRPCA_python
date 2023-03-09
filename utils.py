import numpy as np
from random import choices
import math
import random
from torchvision.datasets import MNIST

def get_swiss(N,sparse_noise_level,gaussian_noise_level=0):
    # Construct swiss roll of dim 20
    P = 20
    t = 4*np.pi*np.random.rand(N,1)
    t = np.array(sorted(t))
    x = (t+1)*np.cos(t)
    y = (t+1)*np.sin(t)
    z = 8*math.pi*np.random.rand(N,1)

    t_max = np.max(t)**2+1
    color = 256/t_max*(t**2+1)
    X = np.concatenate((x,y,z),axis=1)
    for d in range(4,21):
        tmp = t*np.sin(d*t/21).reshape(N,1)
        X = np.concatenate((X,tmp),axis=1)
    clean_X = X.copy()
    # Add sparse noise
    num_outliers = math.ceil(N*P*0.05)
    outlier_loc = choices(list(np.arange(N)),k=num_outliers)
    outlier_dim = choices(list(np.arange(2))+list(np.arange(3,P)),k=num_outliers)
    for r,c in zip(outlier_loc,outlier_dim):
        X[r][c] += np.random.normal(sparse_noise_level,0.03)*int(random.random()<0.5)
    # Add gaussian noise
    X += gaussian_noise_level*np.random.randn(N,20)
    return X,clean_X,color

def get_mnist49(N):
    data = MNIST(root='./data',download=True,train=True)
    indices4 = random.sample([i for i in range(len(data)) if data[i][1] == 4],N//2)
    indices9 = random.sample([i for i in range(len(data)) if data[i][1] == 9],N//2)
    images4 = 1/256*np.array([np.asarray(data[i][0],dtype=float).reshape(784,) for i in indices4])
    images9 = 1/256*np.array([np.asarray(data[i][0],dtype=float).reshape(784,) for i in indices9])
    labels = [0]*len(images4)+[1]*len(images9)
    images = np.concatenate((images4,images9),axis=0)
    return images,labels
    
        
        