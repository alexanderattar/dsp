# Standard Effects Delay Line
# Author: Alexander Attar
# Copyright (C) - Spring 2012

# Effect an audio signal with delayline effects such as Vibrato, Chorus and Flanger. 
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math as m # For math functions
from scikits.audiolab import Sndfile, Format # For reading in audio files
import numpy as np  # For putting audio into arrays
from scipy import signal
import matplotlib.pyplot as plot # to plot output

def main():
    
    # TODO - ADD ERROR CHECKING AT USER INPUTS
    #-------------------PROMPT USER FOR INPUT PARAMETERS---------------------#
    titleOfAudio = raw_input("Enter the title of the audio file: ")
    fb = input("Enter a Feedback value between 0 and 1: ") 
    ff = input("Enter a Feedforward value between 0 and 1: ") 
    bl = input("Enter the amount of Blend between 0 and 1: ") # Blend Parameter
    delayTime = input("Enter the delay time in ms: ")
    delayDepth = input("Enter delay depth in ms: ")
    
     
    # What kind of audio effect are we applying
    effect = raw_input("Effect with noise? Enter yes or no: ")
    
    if effect == 'yes':
        modRate = 0
    elif effect == 'no':
        modRate = input('Enter the modulation rate in Hz: ')
        
    outputTitle = raw_input('Give the output file a title: ')
    
    
    """               Industry Standard Audio Effects:
-----------------------------------------------------------------------------
                BL      FF      FB      DELAY       DEPTH       MOD
Vibrato         0       1       0       0 ms        0-3 ms      0.1-5 Hz Sine
Flanger         0.7     0.7     0.7     0 ms        0-2 ms      0.1-1 Hz Sine
(white)Chorus   0.7     1      -0.7     1-30 ms     1-30 ms     Lowpass noise
Doubling        0.7     0.7     0       10-100 ms   1-100 ms    Lowpass noise

-DAFX pp. 68 - 71
-----------------------------------------------------------------------------
"""
    
    #----------------------------READ AUDIO----------------------------------#
    
    # Audio Data From Sndfile
    f = Sndfile("loop.wav")     # Open File
    fs = f.samplerate           # Find Sample Rate
    channels = f.channels       # Find Channels
    enc = f.encoding            # File encoding
    length = f.nframes          # Find File Length

    #read into a numpy array
    inAudio = f.read_frames(length, dtype=np.float64) 
    # Convert to Mono if file is Stereo
    if channels >= 2: x = (inAudio[:,0] + inAudio[:,1]) / 2.0
    else: x = inAudio

    #-----------------------Define Effect Constants--------------------------#
    DELAY = (((delayTime / 1000.0 * fs)))           # delay in samples
    delayDepth = delayDepth * fs /1000.0
    MOD_RATE = (modRate) / fs                # modRate in samples
    padding = np.zeros(DELAY)                       # padding vector
    x = np.concatenate((x, padding), axis=0)        # Zero pad the input audio
    #------------------------------------------------------------------------#
    
    # Generate lowpass noise
    noise1 = (np.random.random_sample(length))
    # Apply a butter window function
    b,a = signal.butter(2, .001, btype='low', analog = 0, output = 'ba')
    lp = signal.lfilter(b,a, noise1)
        
    # Begin the delay line function
    def delayLine(bl, ff, fb, delayTime, delayDepth, modRate):
        
        # Allocate memory buffers for the output and the modulator
        y = np.zeros((length + DELAY, 1), float)       # for output
        MOD = np.zeros((length + DELAY, 1), float)     # for mod
        delayOut = np.zeros((length + DELAY,1), float) # frac delay out
        
        # Begin Effects Loop
        for n in range(0, length - 1):
            
            # Effect with Noise
            if effect == 'yes':
                            
                MOD[n] = delayDepth/2.0*(lp[n])  + DELAY
                
            # Effect with an LFO
            else:
                MOD[n] = delayDepth/2.0 * (m.sin(2 * m.pi * MOD_RATE * n) + 1) + DELAY
                
            # Calculate the interpolation
            modFloor = m.floor(MOD[n])
            modCeil = m.ceil(MOD[n])
            
            # Find the fractional delay coefficients 
            frac1 = MOD[n] - modFloor
            frac2 = modCeil - MOD[n]
            
            # Apply the fractional delay to the input signal
            delayOut[n] = (frac2 * x[n - modFloor]) + (frac1  * x[n - modCeil])
            
            # Apply feedback/feedforward/blend
            y[n] = ((delayOut[n] * ff) + (x[n] * bl))
            x[n + DELAY] = ((delayOut[n] * fb) + (x[n + DELAY]))
        
        # Normalize the output vector
        maxVal = np.abs(max(y)) 
        minVal = np.abs(min(y)) 
        y = (y / maxVal) *.5 
        y = (y / minVal) *.5
        
        plot.plot(y)
        plot.title("Audio Output")
        plot.xlabel("Time")
        plot.ylabel("Amplitude")
        plot.savefig("test.png")
         
        return y # output the processed audio vector
            
    #----------------------------APPLY DELAY---------------------------------#
    y = delayLine(bl, ff, fb, delayTime, delayDepth, modRate)
    #----------------------------WRITE OUTPUT--------------------------------#
    # Define an output audio format
    formt = Format('wav', 'float64')  
    outFile = Sndfile(outputTitle, 'w', formt, 1, fs)
    outFile.write_frames(y)
    #-----------------------------CLEAN UP-----------------------------------#
    f.close()
    outFile.close()
    #----------------------------END PROGRAM---------------------------------#
    
if __name__ == '__main__':
    main()
    
    
    
    
    
