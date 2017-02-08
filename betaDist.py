import numpy as np
import numpy as np
import pylab as pl
import scipy.special as ss
 
def beta(a, b, mew):
    e1 = ss.gamma(a + b)
    e2 = ss.gamma(a)
    e3 = ss.gamma(b)
    e4 = mew ** (a - 1)
    e5 = (1 - mew) ** (b - 1)
    return (e1/(e2*e3)) * e4 * e5
 
def plot_beta(a, b):
    Ly = []
    Lx = []
    mews = np.mgrid[0:1:100j]
    for mew in mews:
        Lx.append(mew)
        Ly.append(beta(a, b, mew))
    pl.plot(Lx, Ly, label="a=%f, b=%f" %(a,b))
 
def main():
    plot_beta(0.1, 0.1)
    plot_beta(1, 1)
    plot_beta(2, 3)
    plot_beta(8, 4)
    pl.xlim(0.0, 1.0)
    pl.ylim(0.0, 3.0)
    pl.legend()
    pl.show()
 
if __name__ == "__main__":
    main()
