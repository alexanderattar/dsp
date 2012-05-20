# Bark Analyzer
# Author: Alexander Attar
# DSP
# Spring 2012
#
# This script contains a variety of functions for analyzing an audio file.
# Specifically, it allows for the signal to be analyzed on the bark scale
# along with a typical spectogram function. 
# Data visualization is done using the matplotlib library. 

import numpy as np
import struct
import wave
import matplotlib.pyplot as plt

# Distance between 1 and the nearest floating point number
eps = 2.0**(-16.0) # Prevents divide by zero error in numpy.log()

def main():
	
	#-----------------PROMPT USER FOR INPUT PARAMETERS-------------------#
  
	# audioInput = raw_input("Enter the title of audio file: ")
	# winType = raw_input("Select a window type (Hit return for hanning): ")
	# N = input("Input a window size: ")
	# x, fs = wavread(audioInput)
	
	# TEST PARAMS
	N = 1024
	winType = "hanning"
	x,fs = wavread('loop.wav')
	
	# RUN
	window = pickWinType(winType, N)
	hz = spectrogram(x, 1024, fs, 512, winType=None, PLOT=True)
	# getFreqs(x,fs, N=512)
	# plotFreqMags(output)
	freq2bark(hz, PLOT=True)
	
	#-----------------------------END MAIN-------------------------------#
	
def getFreqs(x,fs, N=1028):
	""" Performs the FFT on a signal and retrieves the magnitudes paired 
			with the center frequency of the frequency bin"""

	x	= convertToMono(x)
	x = np.transpose(x)
	
	complexVals = (np.fft.fft(x, axis=0))

	# Initlize Buffers with Zeros
	magnitude = [0] * (len(complexVals) / 2)
	freqs = [0] * (len(complexVals) / 2)
	centerFreqs = [0] * (len(complexVals) / 2)
	
	# Create dictionary to store bin center freqs and magnitude pairs
	bins = {}

	i = 0
	while i < (N / 2 - 1):
		re = complexVals[2 * i] # real values
		im = complexVals[2 * i + 1] # imaginary values
		
		magnitude[i] = abs((np.sqrt(re*re + im*im))) 
		centerFreqs[i] = i * fs / N
		# print centerFreqs[i], magnitude[i]
		bins[centerFreqs[i]] = magnitude[i]
		# print bins[centerFreqs[i]] 
		i += 1
		
	magnitude = np.asarray(magnitude)
	centerFreqs = np.asarray(centerFreqs)
	
	return
		
def spectrogram(x,N,fs,hopsize,winType=None,PLOT=False):
	""" Spectrum analysis. Higher N increases frequency resolution
	 		but lowers time resolution. This function also finds 
	 		approximate frequency values of a signal after taking the
			FFT. Returns an array of frequency values in Hz """

	x = convertToMono(x)

	# No window was selected
	if winType is None:
		win = np.hanning(N / 2 + 1)

	win *= (1.0 / win.sum()) # Normalize window values
	
	pad = np.zeros(N / 2) 
	x = np.concatenate([pad, x, pad])

	# Allocate output List
	output = [] # For the fft values in each window
	hz = [] # For frequencies in hz
	peaks = [] # For the magnitude peaks
	
	n = 0; i = 0
	block = x[:N]
	while len(block) == N:
		output += [np.fft.rfft(block)] # * win # Take the real FFT
		n += hopsize # Shift by the hopsize
		block = x[n : n + N] # from the current idx to the idx plus window size
		
		# Convert each value in the window to approximate frequency in hz
		while i < len(output):
			hz += [fs * abs(output[i]) / N]
			i += 1
		
		# # Optional Peak Picking
		# # Find the peak and interpolate to get a more accurate peak
		# peak = np.argmax(abs(hz))
		# true_peak = parabolic((abs(hz + eps)), peak)[0]
		
		# # Convert to estimated frequency and store data in the list
		# peaks += [fs * true_peak / (N)]
	
	# Convert from Python List to Numpy Array for Matplotlib
	output = np.asarray(output)
	# peaks = np.asarray(peaks)
	hz = np.asarray(hz)
	
	if PLOT:
		fig = plt.figure()
		cax = plt.imshow(np.log(np.abs(output + eps)).transpose(),
		interpolation='nearest',
		aspect='auto',
		origin='bottom')
		ax = fig.gca()
		ax.set_title("Audio Signal\n")
		ax.set_xlabel("Time (windows)")
		ax.set_ylabel("Frequency Bins")
		cbar = fig.colorbar(cax)
		cbar.set_label("dB")
		plt.show()

	return hz
	
def plotFreqMags(output):
	
	fig = plt.figure()
	ax = fig.gca()
	ax.set_title("Magnitude\n")
	ax.set_xlabel("Time (windows)")
	ax.set_ylabel("Frequency")
	plt.plot(output)
	plt.show()
	
def freq2bark(array_of_freqs, PLOT=True):
	""" The Bark scale ranges from 1 to 24 Barks, corresponding to the first 24 
	critical bands of hearing. The published Bark band edges are given in 
	Hertz as [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 
	2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500].
	
	Note that since the Bark scale is defined only up to 15.5 kHz, the highest sampling
	rate for which the Bark scale is defined up to the Nyquist limit, without requiring 
	extrapolation, is 31 kHz. 
	https://ccrma.stanford.edu/~jos/bbt/Bark_Frequency_Scale.html """
	
	# Formula is from:
	# M. R. Schroeder, B. S. Atal, and J. L. Hall. Optimizing digital
	# speech coders by exploiting masking properties of the human ear.
	# J. Acoust Soc Amer, 66 (6): 1979

	g = abs(array_of_freqs + eps)

	# Bark values
	b = 7 * np.log(g / 650 + np.sqrt(1 + (g / 650)**2))
	
	# Critical bandwidth
	c = np.cosh(b / 7) * 650 / 7

	if PLOT:
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		plt.plot(b)
			
		ax1.set_title("Bark Bands")
		ax1.set_ylabel("Bark")
			
		ax2 = fig.add_subplot(212)	
		ax2.set_title("Critical Bandwidth")
		ax2.set_xlabel("Bark\n")
		ax2.set_ylabel("Critical Bandwidth")
			
		plt.plot(b, c)
		
		plt.show()
	
	return b, c
	
#-----------------------------Helper Functions------------------------------#
# Note: wavread and _raw_data helper functions were taken from Eric Humphrey's 
# Python tutorial given on behalf of MARL, Friday 4/27/2012

def wavread(fin):
	""" Read in an Audio file using the wave library """
	wfile = wave.open(fin,'rb')
	x_raw = wfile.readframes(wfile.getnframes())
	x = _rawdata_to_array(x_raw, wfile.getnchannels(), wfile.getsampwidth())
	fs = wfile.getframerate()
	wfile.close()
	return x, float(fs)

def _rawdata_to_array(data, channels, bytedepth):
	"""
	Convert packed byte string into usable numpy arrays
	Returns
	-------
	frame : nd.array of floats
	    array with shape (N,channels), normalized to [-1.0, 1.0]
	"""

	if data is None:
		return None

	N = len(data) / float(channels) / float(bytedepth)
	frame = np.array(struct.unpack('%dh' % N * channels, data)) / (2.0 ** (8 * bytedepth - 1))
	return frame.reshape([N, channels])
	
def pickWinType(winType, N):
	""" Allow the user to pick a window type"""
	# Select window type
	if winType is "bartlett":
		window = np.bartlett(N)
	elif winType is "blackman":
		window = np.blackman(N)
	elif winType is "hamming":
		window = np.hamming(N)
	elif winType is "hanning":
		window = np.hanning(N)
	else:
		window = None
		
		return window
		
# Source of interpolation function - https://gist.github.com/255291
def parabolic(f, x):
	"""Quadratic interpolation for estimating the true position of an
	inter-sample maximum when nearby samples are known.

 	f is a vector and x is an index for that vector.

	Returns (vx, vy), the coordinates of the vertex of a parabola that goes
	through point x and its two neighbors.
	"""
	xv = float(1/2 * (f[x-1] - f[x+1] + 1) / (f[x-1] - 2 * f[x] + f[x+1]) + x)
	yv = float(f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x))
	return (xv, yv)

def split_seq(seq,size):
	""" Split up seq in pieces of size """
	return [seq[i:i+size] for i in range(0, len(seq), size)]

def convertToMono(x):
	""" Take a single channel from a stereo signal """
	if x.ndim == 2: # Stereo
		# Limit to one channel
		x = (x[:,0])
	elif x.ndim == 1: # Mono
		x = x
	else:
		raise ValueError("Input of wrong shape")
	return x
	
def decibels(x):
	""" Return value in decibles """
	return 20.0 * np.log10(x + eps)

def normalize(x):
	""" Normailize values between -1 and 1"""
  return x / np.abs(x).max()
    
if __name__ == '__main__':
    main()