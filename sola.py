# SOLA
# Author: Alexander Attar
# DSP
# Spring 2012

import math as m # For math functions
from scikits.audiolab import Sndfile, Format # For reading in audio files
import numpy as np  # For putting audio into arrays
import matplotlib.pyplot as plot # to plot output

def main():
  
  #-------------------PROMPT USER FOR INPUT PARAMETERS---------------------#
  
  # audioInput = raw_input("Enter audio to time stretch: ")
  # Sa =  input("Analysis hop size Sa in samples   = ")
  # N = input("Analysis block size N in samples  = ")
  # 
  # if Sa > N:
  #    print('Sa must be less than block size !!!')
  # 
  # alpha = input("Time scaling factor alpha: ")
  # L   = input("Overlap in samples (even):      ")
  # Ss = round(Sa * alpha) # Synthesis Step Size
  # if Ss >= N:
  #    print("alpha is not correct, Ss is >= N")
  # elif Ss > (N-L):
  #   print("alpha is not correct, Ss is > N-L")
  # 
  # outputTitle = raw_input("Give the output file a title: ")
  

  # TEST PARAMS
  audioInput = "nocturn.wav"
  Sa = 5000   # analysis step size
  N  = 10000 # Analysis block size N in samples
  L  = 10 # Length of intervals to compare for maximum similarity
  alpha = 0.8  #0.25 <= alpha <= 2, 0.8 - stretch, 1.2 - shrink
  
  f = readAudio(audioInput)
  fs = findSampleRate(f)
  channels = findChannels(f)
  length = findAudioLength(f)
  inAudio = putAudioInNumpy(f, length)
  x = convertToMono(inAudio, channels)
  
  Ss = round(Sa * alpha) # Synthesis Step Size
  M  = int(m.floor(length/Ss)) # Number of Blocks
  
  # Allocate Vectors
  y =  np.zeros(length) # FIX THIS LATER
  head =  np.zeros(length)
  tail =  np.zeros(length)
  for n in range(0, M - 1):
    print n
    
    if (n == 232):
      print "BLAH"
      break
    seg1 =  x[n*Sa : n*Sa + N]
    seg2 = x[(n+ 1) *Sa : (n + 1) * Sa + N]
    
    # Find L1 and L2 for cross correlation comparison
    L1 = seg1[Ss : len(seg1)]
    L2 = seg2[0 : len(L1)]
    
    idx = x_corr(L1, L2)
    
    # Shift segment 2 by the index returned from  
    # the x correlation function
    seg2Shifted = x[(n+ 1) *Ss + idx : (n + 1) * Ss + N + idx]
         
    # The length of the fade out and fade in will vary 
    # per iteration of the loop depending on the amount
    # the segments have to be shifted for maximum correlation.
    fadeLength = N - (Ss + idx)

    # Amplitude of 1 to 0 over number of samples in fade out length
    fadeout = np.linspace(1,0, fadeLength)
    
    # Amplitude of 0 to 1 over number of samples in fade in length
    fadein = np.linspace(0, 1, fadeLength)

    # Split the segments so they can be faded out and in
    # seg1ab = x[n * Ss : ((n + 1) * Ss) + idx]
    # seg1bc = x[(n + 1) * Ss + idx : n * Ss + N]
    # seg2bc = x[(n + 1) * Ss + idx : n * Ss + N]
    # seg2cd = x[n * Ss + N : (n + 1) * Ss + N+idx]
    
    # Different way to index overlapping segments
    seg1ab = seg1[0 : Ss + idx]
    seg1bc = seg1[Ss + idx :]
    seg2bc = seg2Shifted[0 : len(seg1bc)]
    seg2cd = seg2Shifted[len(seg2bc) :]
    
    # Apply the fade out to the segment
    head = seg1bc * fadeout 

    # Apply the fade in to the segment 
    tail =  seg2bc * fadein 
      
    # Overlap Add and put into output vector
    added = head + tail

    y[n * Ss : ((n + 1) * Ss) + idx] =  seg1ab[:]
    y[(n + 1) * Ss + idx : n * Ss + N] = added[:]
    y[n * Ss + N : (n + 1) * Ss + N+idx] = seg2cd[:]
    
  # Plot Waveforms  
  plotAudio(x, "x")
  plotAudio(y, "y")
  plotAudio(head, "head")
  plotAudio(tail, "tail")

  writeAudioOutput(y, fs, f, "test")

def x_corr(L1, L2):
    xcorr = np.correlate(L1, L2, mode='full')
    maxidx = xcorr.argmax()
    shiftingIdx = round(maxidx - len(L1))
    return shiftingIdx
      
def readAudio(audioTitle):
  """ Read an audio file with scikits.audiolab and put into a numpy array """
  # Audio Data From Sndfile
  f = Sndfile(audioTitle)                   # Open File 
  return f
    
def findSampleRate(f): 
  fs = f.samplerate                         # Find Sample Rate 
  return fs
  
def findChannels(f):
  channels = f.channels                     # Find Channels
  return channels 

def findAudioLength(f):
    length = f.nframes                      # Find File Length
    return length
    
def putAudioInNumpy(f, length):
  """Read a numpy array"""
  inAudio = f.read_frames(length, dtype=np.float64)
  return inAudio

def convertToMono(inAudio, channels):
  """Convert file to Mono if in Stereo"""
  if channels >= 2: x = (inAudio[:,0] + inAudio[:,1]) / 2.0
  else: x = inAudio
  return x

def normalize(z):            
    """Normalizes an output vector"""
    normalizedOutput = 0
    maxVal = np.abs(max(z)) 
    minVal = np.abs(min(z))
    normalizedOutput /=  maxVal 
    normalizedOutput /=  minVal
    return normalizedOutput
    
def plotAudio(audioToPlot, title):
  """Plots and audio signal in the time domain"""
  plot.plot(audioToPlot)
  plot.title("Audio Output")
  plot.xlabel("Time")
  plot.ylabel("Amplitude")
  plot.savefig(title + ".png")
  return

def writeAudioOutput(output, fs, f, outputTitle): 
  """Writes audio output"""
  
  outputTitle = "test.wav"
  # Define an output audio format
  formt = Format('wav', 'float64')  
  outFile = Sndfile(outputTitle, 'w', formt, 1, fs)
  outFile.write_frames(output)
  #Clean Up
  f.close()
  outFile.close()
  return

if __name__ == '__main__':
    main()
