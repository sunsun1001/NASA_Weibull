import numpy as np
import numpy as np
import pylab as pl
import scipy.special as ss
import matplotlib.pyplot as plt


x = [1,2,3,4,5,1,2,2,1,3,4,5,2,3,1,4,3,2,1,4]
bin=np.arange(1,6,1)
count,bins,ignored = plt.hist(x,bins=bin)
print count,bins,ignored
print count.max()




plt.show()

