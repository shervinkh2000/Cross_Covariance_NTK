from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
import scipy as sp
from scipy.integrate import quad

######
#Choose the value of l
l = 11




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
    I = quad(integrand1, -np.inf, np.inf, args = (A[n], l))
    integ_0[n] = I[0]
    
legend = ['$\ell$ = ' + str(l)]
plt.plot(A, integ_0)

plt.legend(legend)
plt.xlabel('$y$')
plt.ylabel('$g_\ell(y)$')

plt.show()

plt.close()


    


