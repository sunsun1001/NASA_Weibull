from pylab import *
import numpy as np
import scipy.special as ss
import collections
import matplotlib.pyplot as plt

mdata= np.loadtxt('data.txt', unpack=True, usecols=[0])

mdata.sort()

def weib_hist_plot(data,b):
    plt.hist(data,bins=b)
    show()

plt.hist(mdata,b=xmax,normed=True,alpha=.3)

show()

