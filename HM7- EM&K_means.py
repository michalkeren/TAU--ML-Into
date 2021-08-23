'''
    Michal Keren 204783161
    Itamar Eyal 302309539
'''

import numpy as np
import random
import scipy.stats as scipy
from scipy.spatial import distance
from prettytable import PrettyTable


def gen_sample():
    mean_1 = [1, 1]
    mean_2 = [3, 3]
    Mean = [mean_1, mean_2]
    cov_1 = [[1, 0], [0, 2]]
    cov_2 = [[2, 0], [0, 0.5]]
    Cov= [cov_1, cov_2]
    population= [0,1]
    C= (50,50) #[%]
    i= random.choices(population, weights=C, k=1)[0]
    return np.random.multivariate_normal(Mean[i], Cov[i], 1).T

def P_xj_given(Mean, Cov,xj):
    return scipy.multivariate_normal(Mean, Cov).pdf(xj)

def alpha(X,N,C,Mean,Cov):
    alpha= np.zeros(shape=(2,N))        #init
    for j in range(N):
        xj=X[:,j]
        PDF= [[P_xj_given(Mean[0], Cov[0],xj)],[P_xj_given(Mean[1], Cov[1],xj)]]
        den = np.dot(C,PDF)
        P= PDF/den
        alpha[:, j]= np.multiply(C.reshape(2,1),P).reshape(2)
    #---sanity check--#
    # alpha_1= np.sum(alpha[0,:])
    # alpha_2 = np.sum(alpha[1, :])
    # sum= alpha_1+ alpha_2
    # if (alpha_1+alpha_2)!= N:
    #     print('ERROR')
    # ----------------#
    return alpha


def run_EM(X,N):
    #----init----#
    C= np.random.uniform(0,1,2)
    Mean = [np.random.uniform(0,5,2) ,np.random.uniform(0,5,2)]
    Var = [np.random.uniform(0,5,2) ,np.random.uniform(0,5,2)]
    print(Mean[0])
    t = PrettyTable(['iter', 'C1', 'C2', '\u03BC1', '\u03BC2', '\u03C31.', '\u03C32'])  # table for results
    t.add_row(['init', C[0], C[1], Mean[0], Mean[1], Var[0], Var[1]])
    print(t)
    # -----------#
    for iter in range(100):
        Cov = [[[Var[0][0], 0], [0, Var[0][1]]], [[Var[1][0], 0], [0, Var[1][1]]]]
        Alpha= alpha(X,N,C,Mean,Cov)
        Alpha_l= np.sum(Alpha,axis=1)
        #---
        C= Alpha_l/N
        for l in {0,1}:
            for k in {0,1}:
                Mean[l][k]=np.dot(Alpha[l,:],X[k,:])/Alpha_l[l]
                Var[l][k]= np.dot(Alpha[l,:],(X[k,:]-Mean[l][k])**2)/Alpha_l[l]
        if iter==1 or iter ==9 or iter == 99:
            t.add_row([iter, C[0], C[1],Mean[0],Mean[1],Var[0],Var[1]])
            print([iter, C[0], C[1],Mean[0],Mean[1],Var[0],Var[1]])
    print(t)


def find_nearest_center(centers,x,Ci):
    for j in range(1,len(centers)+1):               # j is the index of a center.
        if distance.euclidean(x, centers[j-1]) < distance.euclidean(x, centers[Ci-1]):
            Ci= j
    return Ci


def run_K_Means(X,N):
    # ----init----#
    Centers= [gen_sample(),gen_sample()]        #(K=2)
    W= random.choices([1,2], weights=(50,50), k=N)
    t = PrettyTable(['K Means iter', '\u03BC1', '\u03BC2'])  # table for results

    # -----------#
    for iter in range(1,101):
        for n in range(N):
            W[n]= find_nearest_center(Centers,X[:,n],W[n])
        for l in {1,2}:
            mask= np.array(W)==l
            count= np.sum(mask)
            Centers[l-1]= np.sum(X[:,mask],1)/count
        if  iter==1 or iter ==9 or iter == 99:
            t.add_row([iter, Centers[0],Centers[1]])
            print([[iter, Centers[0],Centers[1]]])
    print(t)



N=2000
X=np.zeros(shape=(2,N))
for i in range(N):            #generate N sampels.
    X[:,i] = np.array(gen_sample()).reshape((2))

run_EM(X,N)
run_K_Means(X,N)



