#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyleft 2010 Forrest Sheng Bao http://fsbao.net
Project homepage: http://pyeeg.org
"""

from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, ones, log2, std
from numpy.linalg import svd, lstsq
from neurodsp.timefrequency import amp_by_time, freq_by_time
#from neurodsp.laggedcoherence import lagged_coherence as lag_cohe
import numpy as np
from scipy import signal, fftpack
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

def peak_psd(time_series, f_range, N = 1500, fs = 250.0):
    T = 1.0 / fs
    
    yf = fftpack.fft(time_series)
    yff = 2.0/N * np.abs(yf[:N//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    computed_fft = np.array([ xf,yff ])
    indx1  = computed_fft[0,:] > f_range[0] 
    indx2 = computed_fft[0,:] < f_range[1]
    both = indx1*indx2
    selected_band_fft = computed_fft[1,both]

    smooth = savgol_filter(selected_band_fft, 11, 3)

    return smooth.max()
   

def psd(time_series, Fs, f_range):
    f, t, Sxx = signal.spectrogram(time_series, Fs)
        
    indx = np.where(f < f_range[1])[0]
    indx = np.where(f[indx] > f_range[0])[0]
    
    return np.sum(Sxx[indx,:],0).mean()
    
def amplitude_envelope(time_series, Fs, filter_range):
    envelope = amp_by_time(time_series, Fs, filter_range)
    return np.nanmean(envelope)

def instantaneous_frequency(time_series, Fs, filter_range):    
    inst_freq = freq_by_time(time_series, Fs, filter_range)
    return np.nanmean(inst_freq)

#def lagged_coherence(time_series, Fs, filter_range):    
#    #--- lagged_coherence -----
#    lc = lag_cohe(time_series, filter_range, Fs)
#    return np.nanmean(lc)


def higuchi_fd(X,**kwargs):
    """

    Higuchi Fractal Dimension according to:
    T. Higuchi, Approach to an Irregular Time Series on the
    Basis of the Fractal Theory, Physica D, 1988; 31: 277-283.

    Calculate Higuchi Fractal Dimension (HFD) for 1D data/series

    Input:

    X - input (time) series (must be 1D, to be converted into a NumPy array)

    Output:
    
    HFD
    """
    k, L = curve_length(X,**kwargs)
    return lin_fit_hfd(k, L);

def hurst(X):
	""" Compute the Hurst exponent of X. If the output H=0.5,the behavior
	of the time-series is similar to random walk. If H<0.5, the time-series
	cover less "distance" than a random walk, vice verse. 

	Parameters
	----------

	X

		list    
		
		a time series

	Returns
	-------
	H
        
		float    

		Hurst exponent

	Examples
	--------

	>>> import pyeeg
	>>> from numpy.random import randn
	>>> a = randn(4096)
	>>> pyeeg.hurst(a)
	>>> 0.5057444
	
	"""
	
	N = len(X)
    
	T = array([float(i) for i in range(1,N+1)])
	Y = cumsum(X)
	Ave_T = Y/T
	
	S_T = zeros((N))
	R_T = zeros((N))
	for i in range(N):
		S_T[i] = std(X[:i+1])
		X_T = Y - T * Ave_T[i]
		R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])
    
	R_S = R_T / S_T
	R_S = log(R_S)
	n = log(T).reshape(N, 1)
	H = lstsq(n[1:], R_S[1:])[0]
	return H[0]


######################## Begin function definitions ######################
def pfd(X, D=None):
	"""Compute Petrosian Fractal Dimension of a time series from either two 
	cases below:
		1. X, the time series of type list (default)
		2. D, the first order differential sequence of X (if D is provided, 
		   recommended to speed up)

	In case 1, D is computed by first_order_diff(X) function of pyeeg

	To speed up, it is recommended to compute D before calling this function 
	because D may also be used by other functions whereas computing it here 
	again will slow down.
	"""
	if D is None:																						## Xin Liu
		D = first_order_diff(X)
	N_delta= 0; #number of sign changes in derivative of the signal
	for i in range(1,len(D)):
		if D[i]*D[i-1]<0:
			N_delta += 1
	n = len(X)
	return log10(n)/(log10(n)+log10(n/n+0.4*N_delta))


def hjorth_fd(X, Kmax):
	""" Compute Hjorth Fractal Dimension of a time series X, kmax
	 is an HFD parameter
	"""
	L = [];
	x = []
	N = len(X)
	for k in range(1,Kmax):
		Lk = []
		for m in range(0,k):
			Lmk = 0
			for i in range(1,int(floor((N-m)/k))):
				Lmk += abs(X[m+i*k] - X[m+i*k-k])
			Lmk = Lmk*(N - 1)/floor((N - m) / float(k)) / k
			Lk.append(Lmk)
		L.append(log(mean(Lk)))
		x.append([log(float(1) / k), 1])
	
	(p, r1, r2, s)=lstsq(x, L)
	return p[0]

def hjorth(X, D = None):
	""" Compute Hjorth mobility and complexity of a time series from either two 
	cases below:
		1. X, the time series of type list (default)
		2. D, a first order differential sequence of X (if D is provided, 
		   recommended to speed up)

	In case 1, D is computed by first_order_diff(X) function of pyeeg

	Notes
	-----
	To speed up, it is recommended to compute D before calling this function 
	because D may also be used by other functions whereas computing it here 
	again will slow down.

	Parameters
	----------

	X
		list
		
		a time series
	
	D
		list
	
		first order differential sequence of a time series

	Returns
	-------

	As indicated in return line

	Hjorth mobility and complexity

	"""
	
	if D is None:
		D = first_order_diff(X)

	D.insert(0, X[0]) # pad the first difference
	D = array(D)

	n = len(X)

	M2 = float(sum(D ** 2)) / n
	TP = sum(array(X) ** 2)
	M4 = 0;
	for i in range(1, len(D)):
		M4 += (D[i] - D[i - 1]) ** 2
	M4 = M4 / n
	
	return sqrt(M2 / TP), sqrt(float(M4) * TP / M2 / M2)	# Hjorth Mobility and Complexity




def fisher_info(X, Tau, DE, W = None):
	""" Compute Fisher information of a time series from either two cases below:
	1. X, a time series, with lag Tau and embedding dimension DE (default)
	2. W, a list of normalized singular values, i.e., singular spectrum (if W is
	   provided, recommended to speed up.)

	If W is None, the function will do as follows to prepare singular spectrum:

		First, computer an embedding matrix from X, Tau and DE using pyeeg 
		function embed_seq():
			M = embed_seq(X, Tau, DE)

		Second, use scipy.linalg function svd to decompose the embedding matrix 
		M and obtain a list of singular values:
			W = svd(M, compute_uv=0)

		At last, normalize W:
			W /= sum(W)
	
	Parameters
	----------

	X
		list

		a time series. X will be used to build embedding matrix and compute 
		singular values if W or M is not provided.
	Tau
		integer

		the lag or delay when building a embedding sequence. Tau will be used 
		to build embedding matrix and compute singular values if W or M is not
		provided.
	DE
		integer

		the embedding dimension to build an embedding matrix from a given 
		series. DE will be used to build embedding matrix and compute 
		singular values if W or M is not provided.
	W
		list or array

		the set of singular values, i.e., the singular spectrum

	Returns
	-------

	FI
		integer

		Fisher information

	Notes
	-----
	To speed up, it is recommended to compute W before calling this function 
	because W may also be used by other functions whereas computing	it here 
	again will slow down.

	See Also
	--------
	embed_seq : embed a time series into a matrix
	"""

	if W is None:
		M = embed_seq(X, Tau, DE)
		W = svd(M, compute_uv = 0)
		W /= sum(W)	
	
	FI = 0
	for i in range(0, len(W) - 1):	# from 1 to M
		FI += ((W[i +1] - W[i]) ** 2) / (W[i])
	
	return FI




    


def dfa(X, Ave = None, L = None):
	"""Compute Detrended Fluctuation Analysis from a time series X and length of
	boxes L.
	
	The first step to compute DFA is to integrate the signal. Let original seres
	be X= [x(1), x(2), ..., x(N)]. 

	The integrated signal Y = [y(1), y(2), ..., y(N)] is otained as follows
	y(k) = \sum_{i=1}^{k}{x(i)-Ave} where Ave is the mean of X. 

	The second step is to partition/slice/segment the integrated sequence Y into
	boxes. At least two boxes are needed for computing DFA. Box sizes are
	specified by the L argument of this function. By default, it is from 1/5 of
	signal length to one (x-5)-th of the signal length, where x is the nearest 
	power of 2 from the length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...

	In each box, a linear least square fitting is employed on data in the box. 
	Denote the series on fitted line as Yn. Its k-th elements, yn(k), 
	corresponds to y(k).
	
	For fitting in each box, there is a residue, the sum of squares of all 
	offsets, difference between actual points and points on fitted line. 

	F(n) denotes the square root of average total residue in all boxes when box
	length is n, thus
	Total_Residue = \sum_{k=1}^{N}{(y(k)-yn(k))}
	F(n) = \sqrt(Total_Residue/N)

	The computing to F(n) is carried out for every box length n. Therefore, a 
	relationship between n and F(n) can be obtained. In general, F(n) increases
	when n increases.

	Finally, the relationship between F(n) and n is analyzed. A least square 
	fitting is performed between log(F(n)) and log(n). The slope of the fitting 
	line is the DFA value, denoted as Alpha. To white noise, Alpha should be 
	0.5. Higher level of signal complexity is related to higher Alpha.
	
	Parameters
	----------

	X:
		1-D Python list or numpy array
		a time series

	Ave:
		integer, optional
		The average value of the time series

	L:
		1-D Python list of integers
		A list of box size, integers in ascending order

	Returns
	-------
	
	Alpha:
		integer
		the result of DFA analysis, thus the slope of fitting line of log(F(n)) 
		vs. log(n). where n is the 

	Examples
	--------
	>>> import pyeeg
	>>> from numpy.random import randn
	>>> print pyeeg.dfa(randn(4096))
	0.490035110345

	Reference
	---------
	Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling 
	exponents and 	crossover phenomena in nonstationary heartbeat time series. 
	_Chaos_ 1995;5:82-87

	Notes
	-----

	This value depends on the box sizes very much. When the input is a white
	noise, this value should be 0.5. But, some choices on box sizes can lead to
	the value lower or higher than 0.5, e.g. 0.38 or 0.58. 

	Based on many test, I set the box sizes from 1/5 of	signal length to one 
	(x-5)-th of the signal length, where x is the nearest power of 2 from the 
	length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...

	You may generate a list of box sizes and pass in such a list as a parameter.

	"""

	X = array(X)

	if Ave is None:
		Ave = mean(X)

	Y = cumsum(X)
	Y -= Ave

	if L is None:
		L = floor(len(X)*1/(2**array(range(4,int(log2(len(X)))-4))))

	F = zeros(len(L)) # F(n) of different given box length n

	for i in range(0,len(L)):
		n = int(L[i])						# for each box length L[i]
		if n==0:
			print ("time series is too short while the box length is too big")
			print ("abort")
			exit()
		for j in range(0,len(X),n): # for each box
			if j+n < len(X):
				c = range(j,j+n)
				c = vstack([c, ones(n)]).T # coordinates of time in the box
				y = Y[j:j+n]				# the value of data in the box
				F[i] += lstsq(c,y)[1]	# add residue in this box
		F[i] /= ((len(X)/n)*n)
	F = sqrt(F)
	
	Alpha = lstsq(vstack([log(L), ones(len(L))]).T,log(F))[0][0]
	
	return Alpha

#%%           AUXILIAR FUNCTIONS  #############################################
import os
import ctypes
from numpy.ctypeslib import ndpointer

def curve_length(X,opt=False,num_k=50,k_max=None):
    """
    Calculate curve length <Lk> for Higuchi Fractal Dimension (HFD)
    
    Input:
    
    X - input (time) series (must be 1D, to be converted into a NumPy array)
    opt (=True) - optimized? (if libhfd.so was compiled uses the faster code).
    num_k - number of k values to generate.
    k_max - the maximum k (the k array is generated uniformly in log space 
            from 2 to k_max)
    Output:

    k - interval "times", window sizes
    Lk - curve length
    """
    ### Make sure X is a NumPy array with the correct dimension
    X = np.array(X)
    if X.ndim != 1:
        raise ValueError("Input array must be 1D (time series).")
    N = X.size

    ### Get interval "time"
    k_arr = interval_t(N,num_val=num_k,kmax=k_max)

    ### The average length
    Lk = np.empty(k_arr.size,dtype=np.float)

    ### C library
    if opt:
        X = np.require(X, float, ('C', 'A'))
        k_arr = np.require(k_arr, ctypes.c_size_t, ('C', 'A'))
        Lk = np.require(Lk, float, ('C', 'A'))
        ## Load library here
        libhfd = init_lib()
        ## Run the C code here
        libhfd.curve_length(k_arr,k_arr.size,X,N,Lk)
    
    else:
        ### Native Python run
        for i in range(k_arr.size):# over array of k's
            Lmk = 0.0
            for j in range(k_arr[i]):# over m's
                ## Construct X_k^m, i.e. X_(k_arr[i])^j, as X[j::k_arr[i]]
                ## Calculate L_m(k)
                Lmk += (
                    np.sum(
                        np.abs(
                            np.diff( X[j::k_arr[i]] )
                        )
                    )
                    * (N - 1) /
                    (
                        ( (N-j-1)//k_arr[i] )
                        *
                        k_arr[i]
                    )
                ) / k_arr[i]

            ### Calculate the average Lmk
            Lk[i] = Lmk / k_arr[i]

    return (k_arr, Lk);

def lin_fit_hfd(k,L,log=True):
    """
    Calculate Higuchi Fractal Dimension (HFD) by fitting a line to already computed
    interval times k and curve lengths L

    Input:

    k - interval "times", window sizes
    L - curve length
    log (=True) - k and L values will be transformed to np.log2(k) and np.log2(L),
                  respectively

    Output:

    HFD
    """
    if log:
        return (-np.polyfit(np.log2(k),np.log2(L),deg=1)[0]);
    else:
        return (-np.polyfit(k,L,deg=1)[0]);
    
def embed_seq(X,Tau,D):
	"""Build a set of embedding sequences from given time series X with lag Tau
	and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
	i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
	Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding 
	sequence are placed in a matrix Y.

	Parameters
	----------

	X
		list	

		a time series
		
	Tau
		integer

		the lag or delay when building embedding sequence 

	D
		integer

		the embedding dimension

	Returns
	-------

	Y
		2-D list

		embedding matrix built

	Examples
	---------------
	>>> import pyeeg
	>>> a=range(0,9)
	>>> pyeeg.embed_seq(a,1,4)
	array([[ 0.,  1.,  2.,  3.],
	       [ 1.,  2.,  3.,  4.],
	       [ 2.,  3.,  4.,  5.],
	       [ 3.,  4.,  5.,  6.],
	       [ 4.,  5.,  6.,  7.],
	       [ 5.,  6.,  7.,  8.]])
	>>> pyeeg.embed_seq(a,2,3)
	array([[ 0.,  2.,  4.],
	       [ 1.,  3.,  5.],
	       [ 2.,  4.,  6.],
	       [ 3.,  5.,  7.],
	       [ 4.,  6.,  8.]])
	>>> pyeeg.embed_seq(a,4,1)
	array([[ 0.],
	       [ 1.],
	       [ 2.],
	       [ 3.],
	       [ 4.],
	       [ 5.],
	       [ 6.],
	       [ 7.],
	       [ 8.]])

	

	"""
	N =len(X)

	if D * Tau > N:
		print ("Cannot build such a matrix, because D * Tau > N" )
		exit()

	if Tau<1:
		print ("Tau has to be at least 1")
		exit()

	Y=zeros((N - (D - 1) * Tau, D))
	for i in range(0, N - (D - 1) * Tau):
		for j in range(0, D):
			Y[i][j] = X[i + j * Tau]
	return Y


def interval_t(size,num_val=50,kmax=None):
    ### Generate sequence of interval times, k
    if kmax is None:
        k_stop = size//2
    else:
        k_stop = kmax
    if k_stop > size//2:## prohibit going larger than N/2
        k_stop = size//2
        print("Warning: k cannot be longer than N/2")
        
    k = np.logspace(start=np.log2(2),stop=np.log2(k_stop),base=2,num=num_val,dtype=np.int)
    return np.unique(k);

def init_lib():
    libdir = os.path.dirname(__file__)
    libfile = os.path.join(libdir, "libhfd.so")
    lib = ctypes.CDLL(libfile)

    rwptr = ndpointer(float, flags=('C','A','W'))
    rwptr_sizet = ndpointer(ctypes.c_size_t, flags=('C','A','W'))

    lib.curve_length.restype = ctypes.c_int
    lib.curve_length.argtypes = [rwptr_sizet, ctypes.c_size_t, rwptr, ctypes.c_size_t, rwptr]

    return lib;   


def first_order_diff(X):
	""" Compute the first order difference of a time series.

		For a time series X = [x(1), x(2), ... , x(N)], its	first order 
		difference is:
		Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]
		
	"""
	D=[]
	
	for i in range(1,len(X)):
		D.append(X[i]-X[i-1])

	return D

import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y
