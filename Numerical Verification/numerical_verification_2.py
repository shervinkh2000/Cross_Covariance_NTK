from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
import scipy as sp
from scipy.integrate import quad

######
#Choose the value of i
i = 12




def hermit(l, x):

  p = special.hermite(l)

  Hen = 2**(-l/2)*p(x/np.sqrt(2))/np.sqrt(factorial(l))

  return Hen

def integrand1(x, y, l):
  return np.exp(-x**2/2)*np.tanh(y*x)*hermit(l, x)

##number of points to compute
N = 50
##the range of y for which to plot
ymax = 10
ymin = 0.01
A = np.linspace(ymin, ymax, N)


plt.figure()
legend = []

integ_0 = np.zeros(N)

for n in range(N):
    I1 = quad(integrand1, -np.inf, np.inf, args = (A[n], 2*i + 1))
    I2 = quad(integrand1, -np.inf, np.inf, args = (A[n], 1))
    integ_0[n] = np.abs(I1[0]/I2[0])
    
legend = ['$i$ = ' + str(i)]
plt.plot(A, integ_0)

plt.legend(legend)
plt.xlabel('$y$')
plt.ylabel('$g_{2i + 1}(y)/g_1(y)$')

plt.show()

plt.close()


    


