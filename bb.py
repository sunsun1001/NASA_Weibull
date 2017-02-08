from pylab import *
import numpy as np
import scipy.special as ss
import collections
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')



ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('sin', color='r')
plt.show()
