#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

from numpy.fft import fft
from numpy.linalg import svd
import numpy as np

def wavelet_entropy(e, total_e):
    return -(e/total_e)*np.log((e/total_e))


def differential_entropy(signal):
    return (1/2)*np.log(2*np.pi*np.e*np.std(signal)**2)


def modified_samp_entropy(X, M, R):
	N = len(X)

	Em = embed_seq(X, 1, M)	
	Emp = embed_seq(X, 1, M + 1)

	Cm, Cmp = np.zeros(N - M - 1) + 1e-100, np.zeros(N - M - 1) + 1e-100
	# in case there is 0 after counting. Log(0) is undefined.
	for i in range(0, N - M):
		for j in range(i + 1, N - M): 
			Cm[i] += logistic_distance(Em[i], Em[j], R)
			Cmp[i] += logistic_distance(Emp[i], Emp[j], R)

	return  -np.log(sum(Cmp)/sum(Cm))

def rcmse(signal, m, r, tau):
    (Cm, Cmp) = match(coarse_graining(tau, signal[:]), m, r)

    return -np.log(Cmp / Cm)


def renyientropy(px,alpha=1,logbase=2,measure='R'):
    """
    Renyi's generalized entropy

    Parameters
    ----------
    px : array-like
        Discrete probability distribution of random variable X.  Note that
        px is assumed to be a proper probability distribution.
    logbase : int or np.e, optional
        Default is 2 (bits)
    alpha : float or inf
        The order of the entropy.  The default is 1, which in the limit
        is just Shannon's entropy.  2 is Renyi (Collision) entropy.  If
        the string "inf" or numpy.inf is specified the min-entropy is returned.
    measure : str, optional
        The type of entropy measure desired.  'R' returns Renyi entropy
        measure.  'T' returns the Tsallis entropy measure.

    Returns
    -------
    1/(1-alpha)*np.log(sum(px**alpha))

    In the limit as alpha -> 1, Shannon's entropy is returned.

    In the limit as alpha -> inf, min-entropy is returned.
    """
#TODO:finish returns
#TODO:add checks for measure
    if not _isproperdist(px):
        print("px is not a proper probability distribution")
    alpha = float(alpha)


    # gets here if alpha != (1 or inf)
    px = px**alpha
    genent = np.log(px.sum())
    if logbase == 2:
        return 1/(1-alpha) * genent
    else:
        return 1/(1-alpha) * logbasechange(2, logbase) * genent
    
def spectral_entropy(X, Band, Fs, Power_Ratio = None):
	"""Compute spectral entropy of a time series from either two cases below:
	1. X, the time series (default)
	2. Power_Ratio, a list of normalized signal power in a set of frequency 
	bins defined in Band (if Power_Ratio is provided, recommended to speed up)

	In case 1, Power_Ratio is computed by bin_power() function.

	Notes
	-----
	To speed up, it is recommended to compute Power_Ratio before calling this 
	function because it may also be used by other functions whereas computing 
	it here again will slow down.

	Parameters
	----------

	Band
		list

		boundary frequencies (in Hz) of bins. They can be unequal bins, e.g. 
		[0.5,4,7,12,30] which are delta, theta, alpha and beta respectively. 
		You can also use range() function of Python to generate equal bins and 
		pass the generated list to this function.

		Each element of Band is a physical frequency and shall not exceed the 
		Nyquist frequency, i.e., half of sampling frequency. 

 	X
		list

		a 1-D real time series.

	Fs
		integer

		the sampling rate in physical frequency

	Returns
	-------

	As indicated in return line	

	See Also
	--------
	bin_power: pyeeg function that computes spectral power in frequency bins

	"""
	
	if Power_Ratio is None:
		Power, Power_Ratio = bin_power(X, Band, Fs)

	Spectral_Entropy = 0
	for i in range(0, len(Power_Ratio) - 1):
		Spectral_Entropy += Power_Ratio[i] * np.log(Power_Ratio[i])
	Spectral_Entropy /= np.log(len(Power_Ratio))	# to save time, minus one is omitted
	return -1 * Spectral_Entropy

def svd_entropy(X, Tau, DE, W = None):
	"""Compute SVD Entropy from either two cases below:
	1. a time series X, with lag tau and embedding dimension dE (default)
	2. a list, W, of normalized singular values of a matrix (if W is provided,
	recommend to speed up.)

	If W is None, the function will do as follows to prepare singular spectrum:

		First, computer an embedding matrix from X, Tau and DE using pyeeg 
		function embed_seq(): 
					M = embed_seq(X, Tau, DE)

		Second, use scipy.linalg function svd to decompose the embedding matrix 
		M and obtain a list of singular values:
					W = svd(M, compute_uv=0)

		At last, normalize W:
					W /= sum(W)
	
	Notes
	-------------

	To speed up, it is recommended to compute W before calling this function 
	because W may also be used by other functions whereas computing	it here 
	again will slow down.
	"""

	if W is None:
		Y = embed_seq(X, Tau, DE)
		W = svd(Y, compute_uv = 0)
		W /= sum(W) # normalize singular values

	return -1*sum(W * np.log(W))

def ap_entropy(X, M, R):
	"""Computer approximate entropy (ApEN) of series X, specified by M and R.

	Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
	embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of Em 
	is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension are
	1 and M-1 respectively. Such a matrix can be built by calling pyeeg function 
	as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only 
	difference with Em is that the length of each embedding sequence is M + 1

	Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elments 
	are	Em[i][k] and Em[j][k] respectively. The distance between Em[i] and Em[j]
	is defined as 1) the maximum difference of their corresponding scalar 
	components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two 1-D
	vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance between them 
	is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the value of R is
	defined as 20% - 30% of standard deviation of X. 

	Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can 
	check whether Em[j] matches with Em[i]. Denote the number of Em[j],  
	which is in the range of Em[i], as k[i], which is the i-th element of the 
	vector k. The probability that a random row in Em matches Em[i] is 
	\simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1), 
	denoted as Cm[i].

	We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M 
	since the length of each sequence in Emp is M + 1.

	The probability that any two embedding sequences in Em match is then 
	sum(Cm)/ (N - M +1 ). We define Phi_m = sum(np.log(Cm)) / (N - M + 1) and
	Phi_mp = sum(np.log(Cmp)) / (N - M ).

	And the ApEn is defined as Phi_m - Phi_mp.


	Notes
	-----
	
	#. Please be aware that self-match is also counted in ApEn. 
	#. This function now runs very slow. We are still trying to speed it up.

	References
	----------

	Costa M, Goldberger AL, Peng CK, Multiscale entropy analysis of biolgical
	signals, Physical Review E, 71:021906, 2005

	See also
	--------
	samp_entropy: sample entropy of a time series
	
	Notes
	-----
	Extremely slow implementation. Do NOT use if your dataset is not small.

	"""
	N = len(X)

	Em = embed_seq(X, 1, M)	
	Emp = embed_seq(X, 1, M + 1) #	try to only build Emp to save time

	Cm, Cmp = np.zeros(N - M + 1), np.zeros(N - M)
	# in case there is 0 after counting. np.log(0) is undefined.

	for i in range(0, N - M):
#		print i
		for j in range(i, N - M): # start from i, self-match counts in ApEn
#			if max(abs(Em[i]-Em[j])) <= R:# compare N-M scalars in each subseq v 0.01b_r1
			if in_range(Em[i], Em[j], R):
				Cm[i] += 1																						### Xin Liu
				Cm[j] += 1
				if abs(Emp[i][-1] - Emp[j][-1]) <= R: # check last one
					Cmp[i] += 1
					Cmp[j] += 1
		if in_range(Em[i], Em[N-M], R):
			Cm[i] += 1
			Cm[N-M] += 1
		# try to count Cm[j] and Cmp[j] as well here
	
#		if max(abs(Em[N-M]-Em[N-M])) <= R: # index from 0, so N-M+1 is N-M  v 0.01b_r1
#	if in_range(Em[i], Em[N - M], R):  # for Cm, there is one more iteration than Cmp
#			Cm[N - M] += 1 # cross-matches on Cm[N - M]
	
	Cm[N - M] += 1 # Cm[N - M] self-matches
#	import code;code.interact(local=locals())
	Cm /= (N - M +1 )
	Cmp /= ( N - M )
#	import code;code.interact(local=locals())
	Phi_m, Phi_mp = sum(np.log(Cm)),  sum(np.log(Cmp))

	Ap_En = (Phi_m - Phi_mp) / (N - M)

	return Ap_En


def samp_entropy(X, M, R):
	"""Computer sample entropy (SampEn) of series X, specified by M and R.

	SampEn is very close to ApEn. 

	Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
	embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of Em 
	is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension are
	1 and M-1 respectively. Such a matrix can be built by calling pyeeg function 
	as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only 
	difference with Em is that the length of each embedding sequence is M + 1

	Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elments 
	are	Em[i][k] and Em[j][k] respectively. The distance between Em[i] and Em[j]
	is defined as 1) the maximum difference of their corresponding scalar 
	components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two 1-D
	vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance between them 
	is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the value of R is
	defined as 20% - 30% of standard deviation of X. 

	Pick Em[i] as a template, for all j such that 0 < j < N - M , we can 
	check whether Em[j] matches with Em[i]. Denote the number of Em[j],  
	which is in the range of Em[i], as k[i], which is the i-th element of the 
	vector k.

	We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.

	The SampEn is defined as np.log(sum(Cm)/sum(Cmp))

	References
	----------

	Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of biolgical
	signals, Physical Review E, 71:021906, 2005

	See also
	--------
	ap_entropy: approximate entropy of a time series


	Notes
	-----
	Extremely slow computation. Do NOT use if your dataset is not small and you
	are not patient enough.

	"""

	N = len(X)

	Em = embed_seq(X, 1, M)	
	Emp = embed_seq(X, 1, M + 1)

	Cm, Cmp = np.zeros(N - M - 1) + 1e-100, np.zeros(N - M - 1) + 1e-100
	# in case there is 0 after counting. np.log(0) is undefined.

	for i in range(0, N - M):
		for j in range(i + 1, N - M): # no self-match
#			if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1 
			if in_range(Em[i], Em[j], R): # in_range(Em[i], Em[j], R)
				Cm[i] += 1
#			if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
				if abs(Emp[i][-1] - Emp[j][-1]) <= R: # check last one
					Cmp[i] += 1

	Samp_En = np.log(sum(Cm)/sum(Cmp))

	return Samp_En

#%%
    
def logistic_distance(Template, Scroll, Distance):
    m = 0
    for i in range(0,  len(Template)):
        if abs(Template[i] - Scroll[i]) > m:
            m = abs(Template[i] - Scroll[i])
    
    return 1/(1 + np.exp( (m-0.5)/Distance ) )

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

	Y=np.zeros((N - (D - 1) * Tau, D))
	for i in range(0, N - (D - 1) * Tau):
		for j in range(0, D):
			Y[i][j] = X[i + j * Tau]
	return Y

def in_range(Template, Scroll, Distance):
	"""Determines whether one vector is in the range of another vector.
	
	The two vectors should have equal length.
	
	Parameters
	-----------------
	Template
		list
		The template vector, one of two vectors being compared

	Scroll
		list
		The scroll vector, one of the two vectors being compared
		
	D
		float
		Two vectors match if their distance is less than D
		
	Bit
		
	
	Notes
	-------
	The distance between two vectors can be defined as Euclidean distance
	according to some publications.
	
	The two vector should of equal length
	
	"""
	
	for i in range(0,  len(Template)):
			if abs(Template[i] - Scroll[i]) > Distance:
			     return False
	return True
	""" Desperate code, but do not delete
	def bit_in_range(Index): 
		if abs(Scroll[Index] - Template[Bit]) <=  Distance : 
			print "Bit=", Bit, "Scroll[Index]", Scroll[Index], "Template[Bit]",\
			 Template[Bit], "abs(Scroll[Index] - Template[Bit])",\
			 abs(Scroll[Index] - Template[Bit])
			return Index + 1 # move 

	Match_No_Tail = range(0, len(Scroll) - 1) # except the last one 
#	print Match_No_Tail

	# first compare Template[:-2] and Scroll[:-2]

	for Bit in xrange(0, len(Template) - 1): # every bit of Template is in range of Scroll
		Match_No_Tail = filter(bit_in_range, Match_No_Tail)
		print Match_No_Tail
		
	# second and last, check whether Template[-1] is in range of Scroll and 
	#	Scroll[-1] in range of Template

	# 2.1 Check whether Template[-1] is in the range of Scroll
	Bit = - 1
	Match_All =  filter(bit_in_range, Match_No_Tail)
	
	# 2.2 Check whether Scroll[-1] is in the range of Template
	# I just write a  loop for this. 
	for i in Match_All:
		if abs(Scroll[-1] - Template[i] ) <= Distance:
			Match_All.remove(i)
	
	
	return len(Match_All), len(Match_No_Tail)
	"""

def _isproperdist(X):
    """
    Checks to see if `X` is a proper probability distribution
    """
    X = np.asarray(X)
    if not np.allclose(np.sum(X), 1) or not np.all(X>=0) or not np.all(X<=1):
        return False
    else:
        return True


def logbasechange(a,b):
    """
    There is a one-to-one transformation of the entropy value from
    a np.log base b to a np.log base a :

    H_{b}(X)=np.log_{b}(a)[H_{a}(X)]

    Returns
    -------
    np.log_{b}(a)
    """
    return np.log(b)/np.log(a)
   
def match(signal, m, r):
    N = len(signal)

    Em = embed_seq(signal, 1, m)
    Emp = embed_seq(signal, 1, m + 1)

    Cm, Cmp = np.zeros(N - m - 1) + 1e-100, np.zeros(N - m - 1) + 1e-100
    # in case there is 0 after counting. Log(0) is undefined.

    for i in range(0, N - m):
        for j in range(i + 1, N - m):  # no self-match
            # if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1
            if in_range(Em[i], Em[j], r):
                Cm[i] += 1
                # if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                if abs(Emp[i][-1] - Emp[j][-1]) <= r:  # check last one
                    Cmp[i] += 1

    return sum(Cm), sum(Cmp)

## Coarse graining procedure
# tau : scale factor
# signal : original signal
# return the coarse_graining signal
def coarse_graining(tau, signal):
    # signal lenght
    N = len(signal)
    # Coarse_graining signal initialisation
    y = np.zeros(int(len(signal) / tau))
    for j in range(0, int(N / tau)):
        y[j] = sum(signal[i] / tau for i in range(int((j - 1) * tau), int(j * tau)))
    return y

def moving_average(tau, signal):
    N = len(signal)
    aux = np.zeros((N-tau+1,))
    for i in range(N-tau+1):
        aux[i] = np.mean(signal[i:i+tau])
        
    return aux

def bin_power(X,Band,Fs):
	"""Compute power in each frequency bin specified by Band from FFT result of 
	X. By default, X is a real signal. 

	Note
	-----
	A real signal can be synthesized, thus not real.

	Parameters
	-----------

	Band
		list
	
		boundary frequencies (in Hz) of bins. They can be unequal bins, e.g. 
		[0.5,4,7,12,30] which are delta, theta, alpha and beta respectively. 
		You can also use range() function of Python to generate equal bins and 
		pass the generated list to this function.

		Each element of Band is a physical frequency and shall not exceed the 
		Nyquist frequency, i.e., half of sampling frequency. 

 	X
		list
	
		a 1-D real time series.

	Fs
		integer
	
		the sampling rate in physical frequency

	Returns
	-------

	Power
		list
	
		spectral power in each frequency bin.

	Power_ratio
		list

		spectral power in each frequency bin normalized by total power in ALL 
		frequency bins.

	"""

	C = fft(X)
	C = abs(C)
	Power =np.zeros(len(Band)-1);
	for Freq_Index in range(0,len(Band)-1):
		Freq = float(Band[Freq_Index])										## Xin Liu
		Next_Freq = float(Band[Freq_Index+1])
		Power[Freq_Index] = sum(C[np.floor(Freq/Fs*len(X)):np.floor(Next_Freq/Fs*len(X))])
	Power_Ratio = Power/sum(Power)
	return Power, Power_Ratio	