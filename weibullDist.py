import numpy as np
import pylab as pl
import scipy.special as ss
import matplotlib.pyplot as plt
# a=scale parameter b=shape parameter mew=x(variable)

def weib(a,b,mew):
    e1 = (b/a)
    e2 = ((mew/a)**(b-1))
    e3 = np.exp((-(mew/a)**b))
    return e1*e2*e3
 
def plot_weib(a,b,xmin,xmax):
    Ly = []
    Lx = []
    mews = np.mgrid[xmin:xmax:100j]
    for apple in mews:
        Lx.append(apple)
        Ly.append(weib(a, b, apple))
    pl.plot(Lx, Ly, label="a=%f, b=%f" %(a,b))
    
def main():
    xmin=0.0
    xmax=5.0
    a,b = np.loadtxt('data.txt', unpack=True, usecols=[0,1] )
    if type(a)==np.ndarray:
       for k in range(len(a)):
          plot_weib(a[k],b[k])
    elif type(a)==np.float64:
        plot_weib (a,b,xmin,xmax)      
    #scale=input('Please enter the scale parameter: ')
    #shape=input('Please enter the shape parameter: ')    
    #plot_weib(scale,shape) 
    plt.title("This is a PDF Graph")
    plt.xlabel("X")
    plt.ylabel("Probability Density Function")   
    pl.xlim(xmin, xmax)
    pl.ylim(0.0, 3)
    pl.legend()
    pl.show()
    
 
if __name__ == "__main__":
    main()

