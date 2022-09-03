# This function belongs to Piotr Dollar's Toolbox
# http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
# Please refer to the above web page for definitions and clarifications
#
# Calculates the distance between sets of vectors.
#
# Let X be an m-by-p matrix representing m points in p-dimensional space
# and Y be an n-by-p matrix representing another set of points in the same
# space. This function computes the m-by-n distance matrix D where D(i,j)
# is the distance between X(i,:) and Y(j,:).  This function has been
# optimized where possible, with most of the distance computations
# requiring few or no loops.
#
# The metric can be one of the following:
#
# 'euclidean' / 'sqeuclidean':
#   Euclidean / SQUARED Euclidean distance.  Note that 'sqeuclidean'
#   is significantly faster.
#
# 'chisq'
#   The chi-squared distance between two vectors is defined as:
#    d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2;
#   The chi-squared distance is useful when comparing histograms.
#
# 'cosine'
#   Distance is defined as the cosine of the angle between two vectors.
#
# 'emd'
#   Earth Mover's Distance (EMD) between positive vectors (histograms).
#   Note for 1D, with all histograms having equal weight, there is a simple
#   closed form for the calculation of the EMD.  The EMD between histograms
#   x and y is given by the sum(abs(cdf(x)-cdf(y))), where cdf is the
#   cumulative distribution function (computed simply by cumsum).
#
# 'L1'
#   The L1 distance between two vectors is defined as:  sum(abs(x-y));
#
#
# USAGE
#  D = pdist2( X, Y, [metric] )
#
# INPUTS
#  X        - [m x p] matrix of m p-dimensional vectors
#  Y        - [n x p] matrix of n p-dimensional vectors
#  metric   - ['sqeuclidean'], 'chisq', 'cosine', 'emd', 'euclidean', 'L1'
#
# OUTPUTS
#  D        - [m x n] distance matrix
#
# EXAMPLE
#  [X,IDX] = demoGenData(100,0,5,4,10,2,0);
#  D = pdist2( X, X, 'sqeuclidean' );
#  distMatrixShow( D, IDX );
#
# See also PDIST, DISTMATRIXSHOW

# Piotr's Image&Video Toolbox      Version 2.0
# Copyright (C) 2007 Piotr Dollar.  [pdollar-at-caltech.edu]
# Please email me if you find bugs, or have suggestions or questions!
# Licensed under the Lesser GPL [see external/lgpl.txt]

import numpy as np
    
def pdist2(X = None,Y = None,metric = None): 
    if (len(varargin) < 3 or len(metric)==0):
        metric = 0
    
    if np.array([0,'sqeuclidean']) == metric:
        D = distEucSq(X,Y)
    else:
        if 'euclidean' == metric:
            D = np.sqrt(distEucSq(X,Y))
        else:
            if 'L1' == metric:
                D = distL1(X,Y)
            else:
                if 'cosine' == metric:
                    D = distCosine(X,Y)
                else:
                    if 'emd' == metric:
                        D = distEmd(X,Y)
                    else:
                        if 'chisq' == metric:
                            D = distChiSq(X,Y)
                        else:
                            raise Exception(np.array(['pdist2 - unknown metric: ',metric]))
    
    ###########################################################################
    
def distL1(X = None,Y = None): 
    m = X.shape[1-1]
    n = Y.shape[1-1]
    mOnes = np.ones((1,m))
    D = np.zeros((m,n))
    for i in np.arange(1,n+1).reshape(-1):
        yi = Y(i,:)
        yi = yi(mOnes,:)
        D[:,i] = np.sum(np.abs(X - yi), 2-1)
    
    ###########################################################################
    
def distCosine(X = None,Y = None): 
    if (not True  or not True ):
        raise Exception('Inputs must be of type double')
    
    p = X.shape[2-1]
    XX = np.sqrt(np.sum(np.multiply(X,X), 2-1))
    X = X / XX(:,np.ones((1,p)))
    YY = np.sqrt(np.sum(np.multiply(Y,Y), 2-1))
    Y = Y / YY(:,np.ones((1,p)))
    D = 1 - X * np.transpose(Y)
    ###########################################################################
    
def distEmd(X = None,Y = None): 
    Xcdf = cumsum(X,2)
    Ycdf = cumsum(Y,2)
    m = X.shape[1-1]
    n = Y.shape[1-1]
    mOnes = np.ones((1,m))
    D = np.zeros((m,n))
    for i in np.arange(1,n+1).reshape(-1):
        ycdf = Ycdf(i,:)
        ycdfRep = ycdf(mOnes,:)
        D[:,i] = np.sum(np.abs(Xcdf - ycdfRep), 2-1)
    
    ###########################################################################
    
def distChiSq(X = None,Y = None): 
    ### supposedly it's possible to implement this without a loop!
    m = X.shape[1-1]
    n = Y.shape[1-1]
    mOnes = np.ones((1,m))
    D = np.zeros((m,n))
    for i in np.arange(1,n+1).reshape(-1):
        yi = Y(i,:)
        yiRep = yi(mOnes,:)
        s = yiRep + X
        d = yiRep - X
        D[:,i] = np.sum(d ** 2 / (s + eps), 2-1)
    
    D = D / 2
    ###########################################################################
    
def distEucSq(X = None,Y = None): 
    #if( ~isa(X,'double') || ~isa(Y,'double'))
# error( 'Inputs must be of type double'); end;
    m = X.shape[1-1]
    n = Y.shape[1-1]
    #Yt = Y';
    XX = np.sum(np.multiply(X,X), 2-1)
    YY = np.sum(np.multiply(np.transpose(Y),np.transpose(Y)), 1-1)
    D = XX(:,np.ones((1,n))) + YY(np.ones((1,m)),:) - 2 * X * np.transpose(Y)
    ###########################################################################
# function D = distEucSq( X, Y )
#### code from Charles Elkan with variables renamed
# m = size(X,1); n = size(Y,1);
# D = sum(X.^2, 2) * ones(1,n) + ones(m,1) * sum(Y.^2, 2)' - 2.*X*Y';
    
    ### LOOP METHOD - SLOW
# [m p] = size(X);
# [n p] = size(Y);
    
    # D = zeros(m,n);
# onesM = ones(m,1);
# for i=1:n
#   y = Y(i,:);
#   d = X - y(onesM,:);
#   D(:,i) = sum( d.*d, 2 );
# end
    
    ### PARALLEL METHOD THAT IS SUPER SLOW (slower then loop)!
# # From "MATLAB array manipulation tips and tricks" by Peter J. Acklam
# Xb = permute(X, [1 3 2]);
# Yb = permute(Y, [3 1 2]);
# D = sum( (Xb(:,ones(1,n),:) - Yb(ones(1,m),:,:)).^2, 3);
    
    ### USELESS FOR EVEN VERY LARGE ARRAYS X=16000x1000!! and Y=100x1000
# call recursively to save memory
# if( (m+n)*p > 10^5 && (m>1 || n>1))
#   if( m>n )
#     X1 = X(1:floor(end/2),:);
#     X2 = X((floor(end/2)+1):end,:);
#     D1 = distEucSq( X1, Y );
#     D2 = distEucSq( X2, Y );
#     D = cat( 1, D1, D2 );
#   else
#     Y1 = Y(1:floor(end/2),:);
#     Y2 = Y((floor(end/2)+1):end,:);
#     D1 = distEucSq( X, Y1 );
#     D2 = distEucSq( X, Y2 );
#     D = cat( 2, D1, D2 );
#   end
#   return;
# end
    return D