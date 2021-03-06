from pylab import *
import numpy as np
import scipy.special as ss
import collections
import matplotlib.pyplot as plt

def weib_pdf(x,k,l):
    p1 = (k/l)
    p2 = (x/l)**(k-1)
    p3 = np.exp((-(x/l)**k))
    p4 = p1*p2*p3
    return p4

def weib_cdf(x,k,l):
    c1 = np.exp((-(x/l)**k))
    c2 = 1.0 - c1
    return c2


def weib_pdf_plot(xmin,xmax,k,l):
    xdata=np.mgrid[xmin:xmax:60j]
    Ly = []
    Lx = []
    for i in xdata:
        Lx.append(i)
        Ly.append(weib_pdf(i,k,l))

    plt.plot(Lx, Ly)
    show()

def weib_hist_plot(data,b):
    hist(data,bins=10)
    show()

def weib_cdf_plot(xmin,xmax,k,l):
    xdata=np.mgrid[xmin:xmax:60j]
    Ly = []
    Lx = []
    for i in xdata:
        Lx.append(i)
        Ly.append(weib_cdf(i,k,l))
    plt.plot(Lx, Ly)
    show()


def weib_log1(x):
    return np.log(x)

def weib_log2(f):
    np.log(-1.0*np.log(1.0 -f))
    return np.log(-1.0*np.log(1.0 -f))

# data

mdata= np.loadtxt('Winter.txt', unpack=True, usecols=[0])

mdata.sort()
n = len(mdata)
xdata=[]
for i in range(0,n):
    if mdata[i]!=0.0:
       xdata.append(mdata[i])

n = len(xdata)
#print "n", n
xmin = 0
xmax = xdata[n-1] + 5.0

# rank the data
rank=[]
y=collections.Counter(xdata)
z= y.items()
t=z[0:len(y)]
for i in range(0,len(y)):
    a=t[i]
    for k in range(0,a[1]):
        rank.append(i+1)

# find median rank with emprical formula
mrank=[]

for i in rank:
    mean_rank   = (i+0.0)/(n+1.0)
    median_rank = (i-0.3)/(n + 0.4)
    symetrical_cdf = (i-0.5)/(n +0.0)
    parm = median_rank
    mrank.append( parm )

#start Regression

sum1= 0.0
for i in range(1,n+1):
    y = weib_log2(mrank[i-1])
    #print "y=",y
    sum1 = sum1 + y

#print "sum1",sum1

ybar = (1.0/n)*sum1
#print "ybar",ybar

sum2= 00.00e100
for i in range(1,n+1):
    x = weib_log1(xdata[i-1])
    #print "x=",x, "i=",i,xdata[i-1]
    sum2 = sum2 + x
    
#print "sum2", sum2

xbar = (1.0/n)*sum2
#print "xbar",xbar

sum3=0.0
for i in range(1,n+1):
    s = weib_log1(xdata[i-1])*weib_log2(mrank[i-1])
    
    #print "s = " , s
    sum3 = sum3 + s
#print "sum3",sum3

sum4=0.0
for i in range(1,n+1):
    p = (weib_log1(xdata[i-1]))**2
    #print "p = ", p
    sum4 = sum4 + p
#print "sum4",sum4

k=((n*sum3)-(sum1*sum2))/((n*sum4)-(sum2)**2)
print "shape parameter = ",k

l=np.exp(xbar - ybar/k)
print "scale parameter = ",l


sum5=0.0
for i in range(1,n+1):
    u = (weib_log1(xdata[i-1]) -xbar)*(weib_log2(mrank[i-1]) -ybar)
    #print "u = ", u
    sum5 = sum5 + u
#print "sum5",sum5
sum6=0.0
for i in range(1,n+1):
    v = (weib_log1(xdata[i-1]) -xbar)**2
    #print "v = ", v
    sum6 = sum6 + v
#print "sum6",sum6
sum7=0.0
for i in range(1,n+1):
    w = (weib_log2(mrank[i-1]) -ybar)**2
    #print "w = ", w
    sum7 = sum7 + w
#print "sum7",sum7
cc = sum5/((sum6*sum7)**0.5)

print "Correlation Coefficient = ",cc


b = np.log(n)/np.log(2)
yscale = 1
width = (xdata[n-1] - xdata[0])/b


ydata=np.mgrid[xmin:xmax:50000j]
Ly = []
Lx = []
for i in ydata:
    Lx.append(i)
    Ly.append(yscale*weib_pdf(i,k,l))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Lx,Ly, 'b-')
ax1.set_xlabel('Wind Speed')
ax1.set_ylabel('PDF')


ax2 = ax1.twinx()
ax2.hist(xdata,bins=xmax,normed=False)
ax2.set_ylabel('Frequency')

 

show()

 


