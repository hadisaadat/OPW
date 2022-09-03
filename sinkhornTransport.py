import numpy as np
    
def sinkhornTransport(a = None,b = None,K = None,U = None,lambda_ = None,stoppingCriterion = None,p_norm = None,tolerance = None,maxIter = None,VERBOSE = None): 
    # Compute N dual-Sinkhorn divergences (upper bound on the EMD) as well as
# N lower bounds on the EMD for all the pairs
    
    # D= [d(a_1,b_1), d(a_2,b_2), ... , d(a_N,b_N)].
# If needed, the function also outputs diagonal scalings to recover smoothed optimal
# transport between each of the pairs (a_i,b_i).
    
    #---------------------------
# Required Inputs:
#---------------------------
# a is either
#    - a d1 x 1 column vector in the probability simplex (nonnegative,
#    summing to one). This is the [1-vs-N mode]
#    - a d_1 x N matrix, where each column vector is in the probability simplex
#      This is the [N x 1-vs-1 mode]
    
    # b is a d2 x N matrix of N vectors in the probability simplex
    
    # K is a d1 x d2 matrix, equal to exp(-lambda M), where M is the d1 x d2
# matrix of pairwise distances between bins described in a and bins in the b_1,...b_N histograms.
# In the most simple case d_1=d_2 and M is simply a distance matrix (zero
# on the diagonal and such that m_ij < m_ik + m_kj
    
    
    # U = K.*M is a d1 x d2 matrix, pre-stored to speed up the computation of
# the distances.
    
    
    #---------------------------
# Optional Inputs:
#---------------------------
# stoppingCriterion in {'marginalDifference','distanceRelativeDecrease'}
#   - marginalDifference (Default) : checks whether the difference between
#              the marginals of the current optimal transport and the
#              theoretical marginals set by a b_1,...,b_N are satisfied.
#   - distanceRelativeDecrease : only focus on convergence of the vector
#              of distances
    
    # p_norm: parameter in {(1,+infty]} used to compute a stoppingCriterion statistic
# from N numbers (these N numbers might be the 1-norm of marginal
# differences or the vector of distances.
    
    # tolerance : >0 number to test the stoppingCriterion.
    
    # maxIter: maximal number of Sinkhorn fixed point iterations.
    
    # verbose: verbose level. 0 by default.
#---------------------------
# Output
#---------------------------
# D : vector of N dual-sinkhorn divergences, or upper bounds to the EMD.
    
    # L : vector of N lower bounds to the original OT problem, a.k.a EMD. This is computed by using
# the dual variables of the smoothed problem, which, when modified
# adequately, are feasible for the original (non-smoothed) OT dual problem
    
    # u : d1 x N matrix of left scalings
# v : d2 x N matrix of right scalings
    
    # The smoothed optimal transport between (a_i,b_i) can be recovered as
# T_i = diag(u(:,i)) * K * diag(v(:,i));
    
    # or, equivalently and substantially faster:
# T_i = bsxfun(@times,v(:,i)',(bsxfun(@times,u(:,i),K)))
    
    
    # Relevant paper:
# M. Cuturi,
# Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
# Advances in Neural Information Processing Systems (NIPS) 26, 2013
    
    # This code, (c) Marco Cuturi 2013,2014 (see license block below)
# v0.2b corrected a small bug in the definition of the first scaling
# variable u.
# v0.2 numerous improvements, including possibility to compute
#      simultaneously distances between different pairs of points 24/03/14
# v0.1 added lower bound 26/11/13
# v0.0 first version 20/11/2013
    
    # Change log:
# 28/5/14: The initialization of u was u=ones(length(a),size(b,2))/length(a); which does not
#          work when the number of columns of a is larger than the number
#          of lines (i.e. more histograms than dimensions). The correct
#          initialization must use size(a,1) and not its length.
# 24/3/14: Now possible to compute in parallel D(a_i,b_i) instead of being
# limited to D(a,b_i). More optional inputs and better error checking.
# Removed an unfortunate code error where 2 variables had the same name.
    
    # 20/1/14: Another correction at the very end of the script to output weights.
    
    # 15/1/14: Correction when outputting l at the very end of the script. replaced size(b) by size(a).
    
    ## Processing optional inputs
    
    if len(varargin) < 6 or len(stoppingCriterion)==0:
        stoppingCriterion = 'marginalDifference'
    
    if len(varargin) < 7 or len(p_norm)==0:
        p_norm = inf
    
    if len(varargin) < 8 or len(tolerance)==0:
        tolerance = 0.005
    
    if len(varargin) < 9 or len(maxIter)==0:
        maxIter = 5000
    
    if len(varargin) < 10 or len(VERBOSE)==0:
        VERBOSE = 0
    
    ## Checking the type of computation: 1-vs-N points or many pairs.
    
    if a.shape[2-1] == 1:
        ONE_VS_N = True
    else:
        if a.shape[2-1] == b.shape[2-1]:
            ONE_VS_N = False
        else:
            raise Exception('The first parameter a is either a column vector in the probability simplex, or N column vectors in the probability simplex where N is size(b,2)')
    
    ## Checking dimensionality:
    if b.shape[2-1] > b.shape[1-1]:
        BIGN = True
    else:
        BIGN = False
    
    ## Small changes in the 1-vs-N case to go a bit faster.
    if ONE_VS_N:
        I = (a > 0)
        someZeroValues = False
        if not np.all(I) :
            someZeroValues = True
            K = K(I,:)
            U = U(I,:)
            a = a(I)
        ainvK = bsxfun(rdivide,K,a)
    
    ## Fixed point counter
    compt = 0
    ## Initialization of Left scaling Factors, N column vectors.
    u = np.ones((a.shape[1-1],b.shape[2-1])) / a.shape[1-1]
    if str(stoppingCriterion) == str('distanceRelativeDecrease'):
        Dold = np.ones((1,b.shape[2-1]))
    
    ############################### Fixed Point Loop
# The computation below is mostly captured by the repeated iteration of
# u=a./(K*(b./(K'*u)));
    
    # In some cases, this iteration can be sped up further when considering a few
# minor tricks (when computing the distances of 1 histogram vs many,
# ONE_VS_N, or when the number of histograms N is larger than the dimension
# of these histograms).
# We consider such cases below.
    
    while compt < maxIter:

        if ONE_VS_N:
            if BIGN:
                u = 1.0 / (ainvK * (b / (np.transpose(K) * u)))
            else:
                u = 1.0 / (ainvK * (b / np.transpose((np.transpose(u) * K))))
        else:
            if BIGN:
                u = a / (K * (b / np.transpose((np.transpose(u) * K))))
            else:
                u = a / (K * (b / (np.transpose(K) * u)))
        compt = compt + 1
        # check the stopping criterion every 20 fixed point iterations
# or, if that's the case, before the final iteration to store the most
# recent value for the matrix of right scaling factors v.
        if np.mod(compt,20) == 1 or compt == maxIter:
            # split computations to recover right and left scalings.
            if BIGN:
                v = b / (np.transpose(K) * u)
            else:
                v = b / (np.transpose((np.transpose(u) * K)))
            if ONE_VS_N:
                u = 1.0 / (ainvK * v)
            else:
                u = a / (K * v)
            # check stopping criterion
            if 'distanceRelativeDecrease' == stoppingCriterion:
                D = sum(np.multiply(u,(U * v)))
                Criterion = norm(D / Dold - 1,p_norm)
                if Criterion < tolerance or np.isnan(Criterion):
                    break
                Dold = D
            else:
                if 'marginalDifference' == stoppingCriterion:
                    Criterion = norm(sum(np.abs(np.multiply(v,(np.transpose(K) * u)) - b)),p_norm)
                    if Criterion < tolerance or np.isnan(Criterion):
                        break
                else:
                    raise Exception('Stopping Criterion not recognized')
            compt = compt + 1
            if VERBOSE > 0:
                print(np.array(['Iteration :',num2str(compt),' Criterion: ',num2str(Criterion)]))
            if np.any(np.isnan(Criterion)):
                raise Exception('NaN values have appeared during the fixed point iteration. This problem appears because of insufficient machine precision when processing computations with a regularization value of lambda that is too high. Try again with a reduced regularization parameter lambda or with a thresholded metric matrix M.')

    
    if str(stoppingCriterion) == str('marginalDifference'):
        D = sum(np.multiply(u,(U * v)))
    
    if nargout > 1:
        alpha = np.log(u)
        beta = np.log(v)
        beta[beta == - inf] = 0
        if ONE_VS_N:
            L = (np.transpose(a) * alpha + sum(np.multiply(b,beta))) / lambda_
        else:
            alpha[alpha == - inf] = 0
            L = (sum(np.multiply(a,alpha)) + sum(np.multiply(b,beta))) / lambda_
    
    if nargout > 2 and ONE_VS_N and someZeroValues:
        uu = u
        u = np.zeros((len(I),b.shape[2-1]))
        u[I,:] = uu
    
    # ***** BEGIN LICENSE BLOCK *****
#  * Version: MPL 1.1/GPL 2.0/LGPL 2.1
#  *
#  * The contents of this file are subject to the Mozilla Public License Version
#  * 1.1 (the "License"); you may not use this file except in compliance with
#  * the License. You may obtain a copy of the License at
#  * http://www.mozilla.org/MPL/
#  *
#  * Software distributed under the License is distributed on an "AS IS" basis,
#  * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
#  * for the specific language governing rights and limitations under the
#  * License.
#  *
#  * The Original Code is Sinkhorn Transport, (C) 2013, Marco Cuturi
#  *
#  * The Initial Developers of the Original Code is
#  *
#  * Marco Cuturi   mcuturi@i.kyoto-u.ac.jp
#  *
#  * Portions created by the Initial Developers are
#  * Copyright (C) 2013 the Initial Developers. All Rights Reserved.
#  *
#  *
#  ***** END LICENSE BLOCK *****
    return D,L,u,v