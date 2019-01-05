# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:46:12 2017

@author: KimiZ


Maximum Likelihood Estimation Linear Regression

Sources:
   http://jekel.me/2016/Maximum-Likelihood-Linear-Regression/
   http://suriyadeepan.github.io/2017-01-22-mle-linear-regression/

   Data analysis recipes: Fitting a model to data∗  
      David W. Hogg, Jo Bovy, Dustin Lang (2010)
        Equations 9-11
          https://arxiv.org/pdf/1008.4686.pdf

"""


from __future__ import division
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data            = None
File            = '/Users/KimiZ/GRBs2/analysis/Sample/AmatiDataSample.txt'
data            = pd.read_csv(File , sep=',', header=0)

x               = np.log10(data.eiso/(1.0E52))
y               = np.log10(data.epeak)


#   define a function to calculate the log likelihood
def calcLogLikelihood(guess, true, n):
    '''
    This is already set up for the chi-squared as the statistic to minimize.
    
    It is in likelihood format:
    f = ((1./(2*pi* (sig^2)))^(n/2)) * exp(- ( np.dot(error.T, error)/(2 * (sig^2))))
    f is the likelihood funtion
    loglike = log(f)
    
    In log-likelihood format, this would be:
    
    l(theta) = -(n/2)*log(2*pi*(sig^2)) - (1/(2*(sig^2))) * np.dot(error.T, error)
    where error is the error matrix:
    (Y - X * theta).T (Y-X * theta)
    
    '''
    error       = true-guess
    sigma       = np.std(error)
    f           = ((1.0/(2.0*np.pi*sigma*sigma))**(n/2))* \
                    np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)
    
    
#   define my function which will return the objective function to be minimized
def myFunction(parameters):
    m       = parameters[0]
    b       = parameters[1]
    yGuess  = m * x + b
    f       = calcLogLikelihood(yGuess, y, float(len(yGuess)))
    return (-1*f)


#   let's maximize the likelihood (minimize -1*max(likelihood)
res = minimize(myFunction, np.array([0.5,2]), method='L-BFGS-B')
                
res = minimize(LogLikelihood, np.array([1,1]), method='L-BFGS-B')





 # OR IN ONE FUNCTION STEP:               
def LogLikelihood(parameters):
    m           = parameters[0]
    b           = parameters[1]
    ymodel      = m * x + b
    n           = float(len(ymodel))
    error       = y - ymodel  # true y data - model of a line (y - mx - b) or (y - (mx+b))
    sigma       = np.std(error)
    f           = ((1.0/(2.0*np.pi*sigma*sigma))**(n/2))* np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    print(sigma)
    return (-1 * np.log(f) )  # log here is to use log likelihood, f is the likelihood.
#   let's maximize the likelihood (minimize -1*max(likelihood)
res = minimize(LogLikelihood, np.array([0.5,2]), method='L-BFGS-B')

res = minimize(LogLikelihood, np.array([1,1]), method='L-BFGS-B')



 # OR MAKE SIGMA A FREE PARAMETER, AND ONE TO BE RETURNED. 
 # IT'LL GIVE SAME VALUE AS IT WOULD HAVE PRINTED IN THE OTHER FUNCTIONS:               
def LogLikelihood(parameters):
    m           = parameters[0]
    b           = parameters[1]
    sigma       = parameters[2]
    ymodel      = m * x + b
    n           = float(len(ymodel))
    error       = y - ymodel  # true y data - model of a line (y - mx - b) or (y - (mx+b))
    f           = ((1.0/(2.0*np.pi*sigma*sigma))**(n/2))* np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    print(sigma)
    return (-1 * np.log(f) )

#   let's maximize the likelihood (minimize -1*max(likelihood)
res = minimize(LogLikelihood, np.array([0.5,2,0.3]), method='L-BFGS-B')

res = minimize(LogLikelihood, np.array([1,1,1]), method='L-BFGS-B')



# LETS TRY WITHOUT THE DOT PRODUCT, USING THE SUM INSTEAD.
def LogLikelihood(parameters):
    m           = parameters[0]
    b           = parameters[1]
    sigma       = parameters[2]
    ymodel      = m * x + b
    n           = float(len(ymodel))
    f           = ((1.0/(2.0*np.pi*sigma*sigma))**(n/2.)) * np.exp(-1*(( np.sum((y - ymodel) ** 2))/(2*sigma*sigma)))
    print(sigma)
    return (-1 * np.log(f))

#   let's maximize the likelihood (minimize -1*max(likelihood)
res = minimize(LogLikelihood, np.array([0.52,2,0.3]), method='L-BFGS-B')
# THIS GETS US CLOSE, BUT I HAVE TO PROMPT IT WITH PARAMETER VALUES CLOSE TO THE ACTUAL. 
# I WANT TO BE ABLE TO USE INPUTS OF 1,1,1
res = minimize(LogLikelihood, np.array([1,1,1]), method='L-BFGS-B')
# THIS ISN'T WORKING UNLESS I PROVIDE IT WITH VERY SPECIFIC PARAMETERS !





# LETS TRY TAKING THE LOG OF THE LIKELIHOOD FIRST, AND RETURNING THAT AS f.
def LogLikelihood(parameters):
    m           = parameters[0]
    b           = parameters[1]
    sigma       = parameters[2]
    ymodel      = m * x + b
    n           = float(len(ymodel))
    f           = (-0.5*n*np.log(2.0*np.pi*sigma*sigma)) - (np.sum((y-ymodel)**2)/(2.0 * sigma*sigma))
    return (-1*f)

res = minimize(LogLikelihood, np.array([0.52,2,0.3]), method='L-BFGS-B'); res

res = minimize(LogLikelihood, np.array([1,1,1]), method='L-BFGS-B'); res
#   THIS WORKS! USING THE LOGLIKELIHOOD FOR f INSTEAD OF TAKING THE LOG OF THE WHOLE FUNCITON
# AT THE END WORKS BETTER. IT RETURNS VALUES GIVEN ANY PARAMETER INPUTS.









# ANOTHER VERSION OF THE SAME, BUT WRITTEN SLIGHTLY DIFFERENT.
def LogLikelihood(parameters, x, y):
    m, b, sigma       = parameters
    ymodel            = m * x + b
    n                 = float(len(ymodel))
    L   = ((n/2.) * np.log(2.*np.pi*sigma**2) + 1./(2.*sigma**2)*np.sum((y - ymodel)**2))
    # SINCE WE LEFT OFF THE (-) SIGN IN THE EQUATION ABOVE FOR L, WE NEED TO DROP IT 
    # IN THE RETURN AS WELL. 
    return L

res = minimize(LogLikelihood, np.array([0.52,2,0.3]), method='L-BFGS-B', args=(x, y)); res

res = minimize(LogLikelihood, np.array([1,1,1]), method='L-BFGS-B', args=(x, y)); res




# ANOTHER VERSION OF THE SAME, BUT WRITTEN SLIGHTLY DIFFERENT.
# THIS TIME WITH THE DOT PRODUCT AND MATRIX.
def LogLikelihood(parameters, x, y):
    m, b, sigma       = parameters
    ymodel            = m * x + b
    n                 = float(len(ymodel))
    error             = y - ymodel  # true y data - model of a line (y - mx - b) or (y - (mx+b))
    L   = ((n/2.) * np.log(2.*np.pi*sigma**2) + 1./(2.*sigma**2)* np.dot(error.T,error))
    # SINCE WE LEFT OFF THE (-) SIGN IN THE EQUATION ABOVE FOR L, WE NEED TO DROP IT 
    # IN THE RETURN AS WELL. 
    return L

res = minimize(LogLikelihood, np.array([0.52,2,0.3]), method='L-BFGS-B', args=(x, y)); res

res = minimize(LogLikelihood, np.array([1,1,1]), method='L-BFGS-B', args=(x, y)); res



                
# #   perform least squares fit using scikitlearn
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# model = Pipeline([('poly', PolynomialFeatures(degree=2)),
#     ('linear', LinearRegression(fit_intercept=False))])

# model = model.fit(x[:, np.newaxis], y)
# coefs = model.named_steps['linear'].coef_

# #   perform least squares fit using scikitlearn
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# model2 = Pipeline([('linear', LinearRegression(fit_intercept=True, normalize=True))])
# model3 = model2.fit(x[:, np.newaxis], y[:, np.newaxis])
# coefs = model3.named_steps['linear'].coef_
# residuals = model3.named_steps['linear'].residues_






'''

  Now that we have a function that works, we will use it in conjunction with a MCMC technique for generating data.
  We do this to get uncertainty on our slope, y-intercept, and sigma scatter.


'''




#%%





from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import emcee
import corner
#from Amati.tools import *




#   AMATI DATA.
data = None
File            = '/Users/KimiZ/GRBs2/analysis/Sample/AmatiDataSample.txt'
data            = pd.read_csv(File , sep=',', header=0)

xdata           = np.log10(data.eiso/(1.0E52))
ydata           = np.log10(data.epeak)
xdataerr        = np.log10(data.eiso_err/(1.0E52))
ydataerr        = np.log10(data.epeak_err)



# ANOTHER VERSION OF THE SAME, BUT WRITTEN SLIGHTLY DIFFERENT.
# THIS TIME WITH THE DOT PRODUCT AND MATRIX.
def LogLikelihood(parameters, x, y):
    m, b, sigma       = parameters
    ymodel            = m * x + b
    n                 = float(len(ymodel))
    error             = y - ymodel  # true y data - model of a line (y - mx - b) or (y - (mx+b))
    L   = ((n/2.) * np.log(2.*np.pi*sigma**2) + 1./(2.*sigma**2)* np.dot(error.T,error))
    # SINCE WE LEFT OFF THE (-) SIGN IN THE EQUATION ABOVE FOR L, WE NEED TO DROP IT 
    # IN THE RETURN AS WELL. 
    return L

#result = minimize(LogLikelihood, np.array([0.52,2,0.3]), method='L-BFGS-B', args=(x, y)); res

x             = xdata
y             = ydata

result = minimize(LogLikelihood, np.array([1,1,1]), method='L-BFGS-B', args=(x, y)); result
'''
In [3]: result
Out[3]: 
      fun: -14.637374487654924
 hess_inv: <3x3 LbfgsInvHessProduct with dtype=float64>
      jac: array([  5.68434189e-06,   1.42108547e-06,   1.42108547e-06])
  message: 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 112
      nit: 16
   status: 0
  success: True
        x: array([ 0.52089876,  2.05148951,  0.21989444])

'''


# SAVE THE PARAMETERS FOUND BY MAXIMUM LIKELIHOOD. WILL USE THESE TO GENERATE DATA BASED
# OFF OF y = mx + b, WHERE m = m_ml AND b = b_ml, WITH SCATTER f_ml
m_ml, b_ml, f_ml  = result["x"]  # or result.x


# PLOT THE DATA AND MODEL
plt.scatter(x, y)
plt.plot(x, result['x'][0] * x + result['x'][1])
plt.show()


def Prior(parameters):
    m, b, f = parameters
    #if 0.4 < m < 0.6 and 1.0 < b < 3.0 and 0.1 < f < 0.3:
    #if 0.1 < m < 1.0 and 0.1 < b < 5.0 and 0.1 < f < 1.0:
    if -1.0 < m < 1.0 and 0 < b < 10.0 and 0 < f < 1.0:
        return 0.0
    return -np.inf

def Prob(parameters, x, y): 
    lp = Prior(parameters)
    if not np.isfinite(lp):
        return -np.inf
    return lp - LogLikelihood(parameters, x, y) # lp + LogLikelihood(parameters, x, y) # our version needs - sign.


ndim, nwalkers = 3, 100
pos = [result.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, Prob, args=(x, y))
sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))


# WE TAKE THE MAXIMUM LIKELIHOOD RETURNED VALUES TO BE THE TRUE ONES.
m_true, b_true, f_true = result.x

#samples[:, 2] = np.exp(samples[:, 2])  # our f parameter isn't logged.
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

m_mcmc, b_mcmc, f_mcmc



print('%.3f %.3f  %.3f'%(m_mcmc[0], m_mcmc[1], m_mcmc[2]))
print('%.3f %.3f  %.3f'%(b_mcmc[0], b_mcmc[1], b_mcmc[2]))
print('%.3f %.3f  %.3f'%(f_mcmc[0], f_mcmc[1], f_mcmc[2]))

print('%.3f %.3f'%(m_mcmc[0], m_ml))
print('%.3f %.3f'%(b_mcmc[0], b_ml))
print('%.2f %.2f'%(f_mcmc[0], f_ml))

m_err = (m_mcmc[1] + m_mcmc[2])/(2.0)
b_err = (b_mcmc[1] + b_mcmc[2])/(2.0)
f_err = (f_mcmc[1] + f_mcmc[2])/(2.0)



print('\n\n')
print('log(Epk) = b + m log(Eiso) \n\n b = %.3f (+- %.3f) \n m = %.3f (+- %.3f) \n f = %.2f (+- %.2f)'%(b_true,b_err,m_true,m_err,f_true,f_err))
print('  f is the extrinsic scatter and sigma in the parameters. ')







'''

    PLOTS

'''


props = dict(boxstyle=None, facecolor='gainsboro', 
             linewidth=0.0, alpha=0.57)


plt.figure(figsize=(6,5))
xl = np.array([-10, 10])
for m, b, f in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(xl, m * xl + b, color="k", alpha=0.1)
plt.plot(xl, m_true * xl + b_true, color="r", lw=2, alpha=0.8)
plt.errorbar(x, y, xerr=xdataerr, yerr=ydataerr, fmt=".k", 
             capsize=0, alpha=0.2)

#plt.plot(x, 0.9 * x + b_true, color='green')
plt.plot(x, y, ".")
plt.figtext(0.15,0.8,'$%s=%.3f + %.3f %s$'%('\log E^*_{pk}',
                                  b_true, m_true,
                                  ' \ \log E_{iso}'), 
                                    fontsize=18,
                                     bbox=props)                                 
plt.ylim(0,5)
plt.xlim(-5,5)
plt.ylabel('$E^{*}_{pk}$ ($keV$)', fontsize=18)
plt.xlabel('$E_{iso}$ ($erg$)', fontsize=18)
plt.show()
#fig.tight_layout(pad=0.25, w_pad=0, h_pad=0)
#plt.savefig("/Users/KimiZ/GRBs2/python_modeling/plot_Amati_fullsample.png", dpi=250)

#%%

print('%.3f %.3f  %.3f'%(m_mcmc[0], m_mcmc[1], m_mcmc[2]))
print('%.3f %.3f  %.3f'%(b_mcmc[0], b_mcmc[1], b_mcmc[2]))
print('%.3f %.3f  %.3f'%(f_mcmc[0], f_mcmc[1], f_mcmc[2]))

m_err = (m_mcmc[1] + m_mcmc[2])/(2.0)
b_err = (b_mcmc[1] + b_mcmc[2])/(2.0)
f_err = (f_mcmc[1] + f_mcmc[2])/(2.0)



props = dict(boxstyle=None, facecolor='gainsboro', linewidth=0.0, alpha=0.57)
            
label_dict = dict(fontsize=20)
fig = corner.corner(samples, labels=["$m$", "$b$", "$\sigma_{ext}$"], 
                    truth_color='red', 
                    label_kwargs=label_dict,
                    truths=[m_true, b_true, f_true])
                    
plt.figtext(0.5,0.71,'$\log E^*_{pk} \ =\ b + m \ \log E_{iso}$ \n\n $b \ =%.3f(\pm%.3f)$ \n $m=%.3f(\pm%.3f)$ \n $\sigma_{ext}=%.2f (\pm%.2f)$'%(b_true,b_err,m_true,m_err,f_true,f_err),
                                    fontsize = 20,
                                     bbox=props)                                 
                                     
#fig.savefig("/Users/KimiZ/GRBs2/python_modeling/cornerplot_Amati_fullsample.png", dpi=250)


#%%
