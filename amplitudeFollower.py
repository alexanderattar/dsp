# Amplitude Follower
# Author: Alexander Attar
# NYU - DSP
# Spring 2012

import math as m # For math functions
from scikits.audiolab import Sndfile, Format # For reading in audio files
import numpy as np  # For putting audio into arrays
from collections import deque
import matplotlib.pyplot as plot # to plot output
import itertools

def main():

  # =======================================
  # Prompt user for input parameters
  # =======================================

  # audio1 = raw_input("Enter file name of audio to extract envelope from: ")
  # audio2 = raw_input("Enter file name of audio to apply envelope to: ")
  # threshdB = input("Enter the Threshold in dB: ")
  # ratio = input("Enter a compression ratio: ")
  # at = input("Enter the attack time in ms: ")
  # rt = input("Enter the release time in ms: ")
  # scaler = input("Enter a scaler for the envelope follower: ")
  # makeUpGaindB = input("Enter the makeUpGain in dB: ")
  # outputTitle = raw_input("Give the output file a title: ")

  # =======================================
  # Read audio
  # =======================================
  # TEST PARAMS
  audio1 = "percussion.wav"
  audio2 = "noise.wav"
  threshdB = -20.0
  ratio = 4.0
  at = 5.0
  rt = 1000.0
  scaler = 1.0
  makeUpGaindB = 50.0
  outputTitle = "example.wav"

  # Audio Data From Sndfile
  f = Sndfile(audio1)                     # Open File for envFollower
  f2 = Sndfile(audio2)                    # Open File 2 for compressor
  fs = f.samplerate                       # Find Sample Rate of file 1
  fs2 = f2.samplerate                     # Find Sample Rate of file 2
  f_channels = f.channels                 # Find Channels of file 1
  f2_channels = f2.channels               # Find Channels of file 2
  length = f.nframes                      # Find File Length
  length2 = f2.nframes                    # Find File2 Length

  if fs != fs2:
    print "ERROR: USE FILES WITH SAME SAMPLE RATE"
    print "EXITING"
    return

  # Read into a numpy arrays
  inAudio = f.read_frames(length, dtype=np.float64)
  inAudio2 = f2.read_frames(length2, dtype=np.float64)
  # Convert Both Files to Mono if in Stereo
  if f_channels >= 2: xc = (inAudio[:,0] + inAudio[:,1]) / 2.0
  else: xc = inAudio

  if f2_channels >= 2: x = (inAudio2[:,0] + inAudio2[:,1]) / 2.0
  else: x = inAudio2

  # Conversions
  T = 1.0/f.samplerate                              # period
  at /= 1000.0                                      # attack time in seconds
  rt /= 1000.0                                      # release time in seconds
  tav = at + rt / 2
  thresh = 10 **(threshdB / 20)                     # convert back to linear
  makeUpGain = 0.01 * (10 **(makeUpGaindB / 20))    # convert back to linear

  # attack and release in samples
  AT = 1 - m.exp(-2.2 * T / at);
  RT = 1 - m.exp(-2.2 * T / rt);
  TAV = m.exp = m.exp(-1.0 / (fs * tav));

  # If file1 is longer zero pad file2
  if length > length2:
    padding = np.zeros(length - length2)            # padding vector
    x = np.concatenate((x, padding), axis=0)
  # If file2 is longer zero pad file1
  else:
    padding = np.zeros(length2 - length)
    xc = np.concatenate((xc, padding), axis=0)      # Zero pad

  # =======================================
  # Process
  # =======================================
  output = process(xc, x, AT, RT, TAV, length, length2, scaler, thresh, ratio, makeUpGain)

  # =======================================
  # Plot output
  # =======================================
  plotAudio(x, "x")
  plotAudio(xc, "xc")
  plotAudio(output, "output")

  # =======================================
  # Write the audio
  # =======================================
  writeAudioOutput(output, fs, f, f2, outputTitle)


def envelopeFollower(xc, AT, RT, prevG, length, scaler):
  """Follows the amplitude envelope of an audio signal"""

  #-------------------------ENVELOPE DETECTOR---------------------------------#
  xSquared = xc * xc
  #-----------------------------AR AVERAGER-----------------------------------#
  # if input is less than the previous output use attack
  # else use the release
  if xSquared < prevG:
    coeff = AT
  else:
    coeff = RT
  g = (xSquared - prevG)*coeff + prevG

  g = g * scaler

  return g

def compress(x, thresh, ratio, AT, RT, TAV, makeUpGain, prevY, length2):
  """Compresses an audio signal if the envelope is above the threshold"""

  y = 0                                         # for output
  xSquared = 0.0                                  # for rms buffer
  env = 0.0                                     # initialize envelope
  slope = 1 - (1 / ratio)
  #-------------------------ENVELOPE DETECTOR---------------------------------#
  xSquared = (x*x - prevY)*TAV + prevY        # Squarer envelope detector
  #-----------------------------STATIC CURVE----------------------------------#
  if xSquared < 0.000000000000001:
    y = 1
  else:
    y = min(1, (xSquared/(thresh**2))**(-slope/2))

  #----------------------------COMPRESSION------------------------------------#
  # Lowpass filter
  y = (y + prevY) / 2
  y = x * y
  return y

def process(xc, x, AT, RT, TAV, length, length2, scaler, thresh, ratio,
            makeUpGain):
  """
  Processes the audio signals output from the envelope follower
  and the compressor
  """

  prevG = 0
  prevY = 0
  g = np.zeros((length), float)           # for envelope follower output
  y = np.zeros((length2), float)          # for compressor output
  s = np.zeros((length2), float)          # for compressor output
  processed = np.zeros((length2), float)  # for processed output

  windowLength = len(xc)
  # zero pad
  padding = np.zeros(windowLength - length)
  xc = np.concatenate((xc, padding), axis=0)


  for i in range(0, length - 1):

    #----------------------------ENVELOPE FOLLOWER----------------------------#
    # Store the previous sample
    prevG = envelopeFollower(xc[i], AT, RT, prevG, length, scaler)
    #TODO-Somewhere the audio is getting scaled to 0. This is a hack fix for now.
    g[i] = prevG   * 1000.0           # Insert processed sample into a vector

  for n in range(0, length2 - 1):

    #------------------------------COMPRESSION--------------------------------#
    # Store the previous sample
    prevY = compress(x[n], thresh, ratio, AT, RT, TAV, makeUpGain, prevY, length2)
    y[n] = prevY # Insert processed sample into a vector

    # Apply Envelope of Signal1 to Signal2
    processed[n] =  g[n % length] * y[n]# loop envelope over signal1

  return processed

def moving_average(iterable, n = 2000):
    """
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    # http://en.wikipedia.org/wiki/Moving_average
    # from http://docs.python.org/library/collections.html#deque-recipes
    """

    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / float(n)

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

def writeAudioOutput(output, fs, f, f2, outputTitle):
  """Writes audio output"""

  # Define an output audio format
  formt = Format('wav', 'float64')
  outFile = Sndfile(outputTitle, 'w', formt, 1, fs)
  outFile.write_frames(output)

  #Clean Up
  f.close()
  f2.close()
  outFile.close()


if __name__ == '__main__':
    main()