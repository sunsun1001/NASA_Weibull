from pylab import *
import numpy as np
import scipy.special as ss
import scipy.stats as stats
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
 

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


def weib_pdf_plot(xmin,xmax,k,l,scale):
    xdata=np.mgrid[xmin:xmax:60j]
    Ly = []
    Lx = []
    for i in xdata:
        Lx.append(i)
        Ly.append(weib_pdf(i,k,l)*scale)
    
    plt.plot(Lx, Ly)

def weib_hist_plot(data,b):
    hist(data,bins=b)
    show()

def weib_cdf_plot(xmin,xmax,k,l):
    xdata=np.mgrid[xmin:xmax:60j]
    Ly = []
    Lx = []
    for i in xdata:
        Lx.append(i)
        Ly.append(weib_cdf(i,k,l))
    plt.plot(Lx, Ly)


def weib_log1(x):
    return np.log(x)

def weib_log2(f):
    np.log(-1.0*np.log(1.0 -f))
    return np.log(-1.0*np.log(1.0 -f))

# data

if len(sys.argv) != 2:  # the program name and the one arguments
  # stop the program and print an error message
  sys.exit("Must provide Data File Name ")

#Read data from file
mdata= np.loadtxt(sys.argv[1], unpack=True, usecols=[0])


#Print stat about the row data
print "mean=", mdata.mean()," std= ", mdata.std()
print "max=", mdata.max(), "min=",mdata.min()
print "median =" , median(mdata), 
scale_parm2 = stats.scoreatpercentile(mdata, 63)
print " 63 percentile = ",scale_parm2

# ReArrange the data removing any zeros
mdata.sort()
n = len(mdata)
xdata=[]
for i in range(0,n):
    if mdata[i]!=0.0:
       xdata.append(mdata[i])

n = len(xdata)
xmin = xdata[0]
xmax = xdata[n-1] 

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
    

#for i in rank:
#    print rank[i-1], "  ",xdata[i-1], "  ", mrank[i-1] , "  ", weib_log2(mrank[i-1]),"  ", weib_log1(xdata[i-1])

Ly = []
Lx = []


for i in rank:
    Lx.append(weib_log1(xdata[i-1]))
    Ly.append(weib_log2(mrank[i-1]))

sum1 = 0.0
sum2 = 0.0
for i in Lx:
    sum1 = sum1 + i*i
    sum2 = sum2 + i


n = len(Lx)
D = sum1 - 1./n * sum2*sum2
x_bar = mean(Lx)
p_coeff, residuals, _, _, _ = polyfit(Lx, Ly, 1, full=True)
shape_parm = p_coeff[0]
print "slope = ", p_coeff[0], " intercept= ", p_coeff[1]
scale_parm = np.exp(-1.0*p_coeff[1]/p_coeff[0])
print "shape parm= ",shape_parm, " scale parm =" , scale_parm 
dm_squared = 1./(n-2)*residuals/D 
dc_squared = 1./(n-2)*(D/n + x_bar**2)*residuals/D

print "Error in slope = ", dm_squared , " Error in intercept = ", dm_squared
    
# Plot data points 
plot( Lx,Ly, 'kx', markersize=2, label='data' ) 

# Find and plot 1st order line of best fit 
coeff = polyfit( Lx,Ly, 1, )
p = poly1d( coeff ) 
x=linspace( min(Lx), max(Lx), 100 )

plot( x, p(x), label='Best Fit Line' )
# Add titles/labels and a legend to the graph title( 'Best Fit of Experimental Data' )
xlabel( 'weib_log1' )
ylabel( 'weib_log2' ) 
legend( loc='best' )

show()  

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
shape_parm1=((n*sum3)-(sum1*sum2))/((n*sum4)-(sum2)**2)
print "shape parameter1 = ",shape_parm1

scale_parm1 =np.exp(xbar - ybar/shape_parm1)
print "scale parameter1 = ",scale_parm1


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



# Plot a histogram of data
pdf = PdfPages(sys.argv[1] + '.pdf')
xdata_n = xdata/sum(xdata)
count, bins, ignored = plt.hist(xdata,bins=xmax)

spp = ((shape_parm1+shape_parm)/2.0)
slp = scale_parm2
print type(spp), type(slp)
print count.max()
print weib_pdf(xdata,spp,slp).max()
scale_factor = count.max()/weib_pdf(xdata,spp,slp).max()

print "scale_factor = " ,scale_factor

weib_pdf_plot(xmin,xmax,spp,slp,scale_factor)
plt.title(sys.argv[1])
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Frequency')
#plt.ylim((0,800))
#plt.show()
plt.savefig(pdf, format='pdf')
close()
d = pdf.infodict()
d['Title'] = 'WeibWeibull distribution'  
d['Author'] = u'Sunny Omparkash Singh'
d['Subject'] = 'WeibWeibull distribution '
d['Keywords'] = 'PdfPages Title Subject Author'
d['CreationDate'] = datetime.datetime.today()
d['ModDate'] = datetime.datetime.today()
pdf.close()
