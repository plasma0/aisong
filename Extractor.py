import essentia
import essentia.standard
import essentia.streaming
import IPython
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import numpy as np
import mpmath
from keras.models import Sequential
from keras.layers import Dense



def ix(ar):
    i=0
    s=[]
    for i in ar:
        s.append(i)
        i += 1
    return s

loader = essentia.standard.MonoLoader(filename='./electro.mp3')
audio = loader()
frame = audio[5*44100 : 6*44100 + 1024]
spectrum = essentia.standard.Spectrum()
w = essentia.standard.Windowing(type = 'hann')
fft = essentia.standard.FFT()
spec = fft(frame)

X = [x.real for x in spec]
Y = [x.imag for x in spec]
Z = ix(spec)

model = Sequential()

model.add(Dense(units=1800), activation='relu', input_dim=(len(X)*len(Y)*len(Z)))
model.add(Dense(units=180), activation = 'softmax')

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
