import numpy as np
    
def OPW_w(X = None,Y = None,a = None,b = None,lamda1 = None,lamda2 = None,delta = None,VERBOSE = None): 
    # Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences
# X and Y
    
    # -------------
# INPUT:
# -------------
# X: a N * d matrix, representing the input sequence consists of of N
# d-dimensional vectors, where N is the number of instances (vectors) in X,
# and d is the dimensionality of instances;
# Y: a M * d matrix, representing the input sequence consists of of N
# d-dimensional vectors, , where N is the number of instances (vectors) in
# Y, and d is the dimensionality of instances;
# iterations = total number of iterations
# a: a N * 1 weight vector for vectors in X, default uniform weights if input []
# b: a M * 1 weight vector for vectors in Y, default uniform weights if input []
# lamda1: the weight of the IDM regularization, default value: 50
# lamda2: the weight of the KL-divergence regularization, default value:
# 0.1
# delta: the parameter of the prior Gaussian distribution, default value: 1
# VERBOSE: whether display the iteration status, default value: 0 (not display)
    
    # -------------
# OUTPUT
# -------------
# dis: the OPW distance between X and Y
# T: the learned transport between X and Y, which is a N*M matrix
    
    # -------------
# c : barycenter according to weights
# ADVICE: divide M by median(M) to have a natural scale
# for lambda
    
    # -------------
# Copyright (c) 2017 Bing Su, Gang Hua
# -------------
    
    # -------------
# License
# The code can be used for research purposes only.
    
    if len(varargin) < 3 or len(lamda1)==0:
        lamda1 = 50
    
    if len(varargin) < 4 or len(lamda2)==0:
        lamda2 = 0.1
    
    if len(varargin) < 5 or len(delta)==0:
        delta = 1
    
    if len(varargin) < 6 or len(VERBOSE)==0:
        VERBOSE = 0
    
    tolerance = 0.005
    maxIter = 20
    # The maximum number of iterations; with a default small value, the
# tolerance and VERBOSE may not be used;
# Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
# transport;
    p_norm = inf
    N = X.shape[1-1]
    M = Y.shape[1-1]
    dim = X.shape[2-1]
    if Y.shape[2-1] != dim:
        print('The dimensions of instances in the input sequences must be the same!')
    
    P = np.zeros((N,M))
    mid_para = np.sqrt((1 / (N ** 2) + 1 / (M ** 2)))
    for i in np.arange(1,N+1).reshape(-1):
        for j in np.arange(1,M+1).reshape(-1):
            d = np.abs(i / N - j / M) / mid_para
            P[i,j] = np.exp(- d ** 2 / (2 * delta ** 2)) / (delta * np.sqrt(2 * np.pi))
    
    #D = zeros(N,M);
    S = np.zeros((N,M))
    for i in np.arange(1,N+1).reshape(-1):
        for j in np.arange(1,M+1).reshape(-1):
            #D(i,j) = sum((X(i,:)-Y(j,:)).^2);
            S[i,j] = lamda1 / ((i / N - j / M) ** 2 + 1)
    
    D = pdist2(X,Y,'sqeuclidean')
    #D = D/(10^2);
# In cases the instances in sequences are not normalized and/or are very
# high-dimensional, the matrix D can be normalized or scaled as follows:
# D = D/max(max(D));  D = D/(10^2);
    
    K = np.multiply(P,np.exp((S - D) / lamda2))
    # With some parameters, some entries of K may exceed the maching-precision
# limit; in such cases, you may need to adjust the parameters, and/or
# normalize the input features in sequences or the matrix D; Please see the
# paper for details.
# In practical situations it might be a good idea to do the following:
# K(K<1e-100)=1e-100;
    
    if len(a)==0:
        a = np.ones((N,1)) / N
    
    if len(b)==0:
        b = np.ones((M,1)) / M
    
    ainvK = bsxfun(rdivide,K,a)
    compt = 0
    u = np.ones((N,1)) / N
    # The Sinkhorn's fixed point iteration
# This part of code is adopted from the code "sinkhornTransport.m" by Marco
# Cuturi; website: http://marcocuturi.net/SI.html
# Relevant paper:
# M. Cuturi,
# Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
# Advances in Neural Information Processing Systems (NIPS) 26, 2013
    while compt < maxIter:

        u = 1.0 / (ainvK * (b / (np.transpose(K) * u)))
        compt = compt + 1
        # check the stopping criterion every 20 fixed point iterations
        if np.mod(compt,20) == 1 or compt == maxIter:
            # split computations to recover right and left scalings.
            v = b / (np.transpose(K) * u)
            u = 1.0 / (ainvK * v)
            Criterion = norm(sum(np.abs(np.multiply(v,(np.transpose(K) * u)) - b)),p_norm)
            if Criterion < tolerance or np.isnan(Criterion):
                break
            compt = compt + 1
            if VERBOSE > 0:
                print(np.array(['Iteration :',num2str(compt),' Criterion: ',num2str(Criterion)]))

    
    U = np.multiply(K,D)
    dis = sum(np.multiply(u,(U * v)))
    T = bsxfun(times,np.transpose(v),(bsxfun(times,u,K)))
    return dis,T
    
    return dis,T