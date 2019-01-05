'''

This file contains all of the model commands and setup of the plotting commands.

********** mpmath.gammainc is the only one that works when alpha+1. is 0.0, i.e. when alpha = -1.0

MATH CONSIDERATIONS:
The mpmath module will be used for the incomplete gamma function (gammainc) found in the 'copl' and 'band' models.  and hypergeometric function (mpmath.hyp2f1).

The Gauss hypergeometric function:

	mpmath:
		mpmath.hyp2f1(a,b,c,z) is equivalent to hyper([a,b],[c],z). This version requires float() to surround it. float(mpmath.hyp2f1(a,b,c,z))
			Link:  http://docs.sympy.org/0.7.1/modules/mpmath/functions/hypergeometric.html
			
	scipy.special:
		scipy.special.hyp2f1(a,b,c,z) does not need float() around it.  This one has issues with certain values of a,b,c&z. If the function doesn't seem to work for a burst, use the mpmath version. ** See Warning on computation time below.
			Link:  http://docs.scipy.org/doc/scipy-0.17.1/reference/generated/scipy.special.hyp2f1.html#scipy.special.hyp2f1

	

The Incomplete Gamma Function:

	mpmath:
	
		mpmath.gammainc is used for the incomplete gamma function instead of scipy.special.gammainc or scipy.special.gammaincc, where gammaincc = 1-gammainc . --inc is integrated from 0 to x and incc is integrated from x to inf.
		Neither of scipy gammainc functions can handle negative numbers for a or b, where gammainc(a,b) and mpmath's function can.  Also, I checked the gammainc answers with wolfram alpha and scipy's gammainc answers are always wrong where mpmath's are consistent.

	
*** WARNING ON COMPUTATION TIME FOR MPMATH FUNCTIONS:
		mpmath functions require a lot of computation time and for functions that don't utilize gammainc or mpmath's hyp2f1, computation time is comparable to C++ or Fortran XSPEC models.  
			* Python Models that utilize these functions take a lot more time, thus, the 'grbm' XSPEC model is preferred over this Python 'band' model since both are written the same and normalized at 100.0 keV.
			* The XSPEC 'cutoffpl' model is normalized to 1.0 keV, making the Python 'copl' model preferable over the XSPEC model.  There is no smoothly broken power-law in XSPEC.
	
Other Math Functions:
	* Math or Numpy versions of exp, log, log10, cosh, and sinh can be used. For sech(x), 1./cosh(x) is used.
	* mpmath's log and log10 are ln and log or log(1,10) for log base 10.
	* math.pow(2,3) vs 2**3 computation time was tested and ** beats math.pow().  The math and numpy functions are comparable in computation time.


## Power law with high energy exponential cutoff.
## Number of model parameters: 2
##   1	   photIdx		 powerlaw photon index
##   2	   cutoff		  energy of exponential cutoff (in
##						   energy units, e.g. keV). if <= 0
##						   then no cutoff applied.
## Intrinsic energy range:
##   Emin = epsilon(>0), Emax = infinity
##
## algorithm:
##   n(E)= E**(-photIdx) * exp(-E/cutoff) dE
##   This relies on an approximate incomplete gamma function
##   calculation to perform the integral over n(E).  
##   WARNING: The approximation loses accuracy in the region,
##   10^-6 > (1-photIdx) > 0.
'''

from __future__ import division
from xspec import *
import mpmath
from mpmath import polylog
from numpy import log, log10, exp, cosh, sinh
import scipy.special
from scipy.special import hyp2f1
#from mpmath import gammainc
#from mpmath import sech, hyp2f1
#from sympy import lowergamma, uppergamma
#import sympy

from mpmath import fp



########################################################################
## SIMPLE POWER-LAW normalized at 100 keV
def lpow(engs, params, flux):
	plIdx = float(params[0])
	
	for i in range(len(engs)-1):
	
		multiplier		= (-(100**(-plIdx)))
		lowIntegral		= ((engs[i]**(plIdx+1.)) /(plIdx + 1.))
		highIntegral	= ((engs[i+1]**(plIdx+1.))/(plIdx + 1.))
		
		val = multiplier * (lowIntegral - highIntegral)
		flux[i] = val

lpowInfo = ("plIndex NONE -2.0 -4. -4. 1.0 1.0 0.001",)
AllModels.addPyMod(lpow, lpowInfo, 'add')



########################################################################
## CUTOFF POWER-LAW normalized at 100 keV
'''
Algorithm:
n(E)= A * (E**(-cpl_Idx) * exp(-E/cutoff)) dE

fp.gammainc( cpl_Idx+1., a = (engs[i]/cutoff) )
# a = is the lower integral limit.
# b would be the upper integral limit, which is infinity by default.  The default is used when b is not set.
'''
# SEE THIS WEBSITE ABOUT SPEEDING UP MPMATH COMPUTATION TIME USING fp.gammainc instead of gammainc
def copl(engs, params, flux):

	for i in range(len(engs)-1):
	
		cutoff		= float(params[1])
		cpl_Idx		= float(params[0])
		
		multiplier		= (100**(-cpl_Idx))*(cutoff**(cpl_Idx+1.))
		
		lowIntegral		= float(fp.gammainc(cpl_Idx+1., a=(engs[i]/cutoff))) # a - lower integ. limit. b-upper is inf.
		
		highIntegral	= float(fp.gammainc(cpl_Idx+1., a=(engs[i+1]/cutoff)))
		
		val			=  multiplier * (lowIntegral - highIntegral)
		flux[i]		= val

coplInfo = ("cplIndex NONE -1.0 -2.0 -1.8 0.8 1.0 0.01",
			"highEcut NONE 300. 10. 30. 9000. 10000. 3.0",) # 6000, 7000 are old values.
AllModels.addPyMod(copl, coplInfo, 'add')




########################################################################
## BAND FUNCTION normalized at 100 keV
'''
Band, D., Matteson, J., Ford, L., et al. 1993, ApJ, 413, 281
Algorithm:
for E < ((alpha-beta)*efold):
	n(E)= A * ( ((E/100)**alpha) * exp(-E/efold)) dE
for E >= ((alpha-beta)*efold):
	n(E)= A * ( (((alpha-beta)* efold)/100)**(alpha-beta) * (E/100)**beta * exp(-(alpha-beta))) dE

'''
def band(engs, params, flux):
	alpha = float(params[0]) # low-energy index
	beta = float(params[1]) # high-energy index
	efold = float(params[2]) # e-folding energy, E_0
	
	for i in range(len(engs)-1):
		if engs[i] < ((alpha - beta) * efold):
			multiplier = ((100**(-alpha)) * (efold**(alpha+1.)))
			lowIntegral =  float(mpmath.gammainc(alpha + 1., (engs[i]/efold)))
			highIntegral = float(mpmath.gammainc(alpha + 1., (engs[i+1]/efold)))
			val = multiplier * (lowIntegral - highIntegral)
			flux[i] = val
		
		else:
			multiplier = ((1./(beta + 1.))* ((100**(-alpha)) * ((alpha - beta)**(alpha-beta)) * -exp(beta-alpha) * (efold**(alpha-beta))))
			lowIntegral = engs[i]**(beta+1.)
			highIntegral = engs[i+1]**(beta+1.)
			val = multiplier * (lowIntegral - highIntegral)
			flux[i] = val


bandInfo = ("alpha NONE -1.0 -1.889 -1.8 2.0 5.0 0.01",
			"beta NONE -2.0 -8.0 -5.0 -1.9 -1.9 0.02",
			"ecut keV 300. 10. 30. 3000. 7000. 3.0",)
AllModels.addPyMod(band, bandInfo, 'add')



########################################################################
## SBPL FUNCTION normalized at 100 keV, hidden within the constants.
'''
F. Ryde 1998:  http://arxiv.org/abs/astro-ph/9811462v1
Equation 2.
Algorithm:
a = (IdxHig - IdxLow)/2.
b = (IdxLow + IdxHig)/2.
n(E)= A * ( (E/100)**b * ((cosh(log10(E/ebk)/delta) / cosh(log10(100/ebk)/delta) )**(a*delta*log(10)))) dE

'''
def sbpl(engs, params, flux):
    
    a  = float(params[0])  # alpha 
    b  = float(params[1])  # beta
    k  = float(params[2])  # ebreak
    d  = 0.3               # break scale.  delta

    for i in range(len(engs)-1):

        p1 = (b - a)/2.
        p2 = (a + b)/2.

        # LOG AND LOG10 CONSTANTS TO SIMPLIFY THE FUNCTION
        c1 = 4.60517   # log(100)
        c2 = 0.868589  # 2*(1./log(10)) see also logarithmic properties of log((1+n)/(1-n))
        c3 = 2.30259   # log(10)
        c4 = 0.434294  # 1./log(10)
        c5 = 1.59603   # log(2)*log(10) = log(2)+log(10)
        c6 = 1.15129   # log(10)/2.
 

        multiplier = ( (1./(b+1.)) * -exp(-c1*p2) * (cosh(c5*p1*d)-sinh(c5*p1*d)))

        lowIntegral = (
            engs[i]**(p2+1)
            * ( ((engs[i]/k)**(-c2/d) +1.)**(-c3*p1*d))
            * ((((engs[i]/k)**(-c4/d)) + ((engs[i]/k)**(c4/d)))**(c3*p1*d))
            * float(hyp2f1(-c3*p1*d, -c6*d*(b+1.), -c6*d*(b+1.)+1., -(engs[i]/k)**(-c2/d)))
            * (cosh((c4 * log(engs[i]/k))/d)**(-c3*p1*d))
            * (((1./cosh((c4*(log(1/k)+c1))/d)) * cosh((c4*log(engs[i]/k))/d))**(c3*p1*d))
            )

        highIntegral = (
            engs[i+1]**(p2+1)
            * ( ((engs[i+1]/k)**(-c2/d) +1.)**(-c3*p1*d))
            * ((((engs[i+1]/k)**(-c4/d)) + ((engs[i+1]/k)**(c4/d)))**(c3*p1*d))
            * float(hyp2f1(-c3*p1*d, -c6*d*(b+1.), -c6*d*(b+1.)+1., -(engs[i+1]/k)**(-c2/d)))
            * (cosh((c4 * log(engs[i+1]/k))/d)**(-c3*p1*d))
            * (((1./cosh((c4*(log(1/k)+c1))/d)) * cosh((c4*log(engs[i+1]/k))/d))**(c3*p1*d))
            )


        val         = multiplier * (lowIntegral - highIntegral)
        flux[i]     = val


sbplInfo = ("alpha NONE -1.0 -2.0 -1.8 0.8 1.0 0.01",
			"beta NONE -2.0 -5.0 -4.8 -1.0 0.0 0.01",
			"ebreak keV 300. 10. 30. 6000. 7000. 3.0",)
			#"delta NONE 0.3 0.0 0.0 10.0 10.0 0.01",)
AllModels.addPyMod(sbpl, sbplInfo, 'add')




########################################################################
THIS VERSION IS THE ORIGIANL USED ON THE FITS.  IT SHOULD BE THE EXACT
SAME AS THE ONE CURRENTLY IN USE.  IT'S JUST THAT CONSTANTS ARE NAMED
DIFFERENTLY.
########################################################################
## SBPL FUNCTION normalized at 100 keV, hidden within the constants.
'''
F. Ryde 1998:  http://arxiv.org/abs/astro-ph/9811462v1
Equation 2.
Algorithm:
a = (IdxHig - IdxLow)/2.
b = (IdxLow + IdxHig)/2.
n(E)= A * ( (E/100)**b * ((cosh(log10(E/ebk)/delta) / cosh(log10(100/ebk)/delta) )**(a*delta*log(10)))) dE

'''
def sbpl(engs, params, flux):
	IdxLow = float(params[0]) # low-energy index, aka lambda_1 or alpha
	IdxHig = float(params[1]) # high-energy index, aka lambda_2 or beta
	ebk = float(params[2]) # Ebreak, break energy
	delta = 0.3 # breakscale energy in keV. Uncomment the next line to use as a fit parameter.
	#delta = float(params[3])
	for i in range(len(engs)-1):
		delta = delta
		a = (IdxHig - IdxLow)/2.
		b = (IdxLow + IdxHig)/2.
		# LOG AND LOG10 CONSTANTS TO SIMPLIFY THE FUNCTION
		c1 = 4.60517 # log(100)
		c2 = 0.868589 # 2*(1./log(10)) see also logarithmic properties of log((1+n)/(1-n))
		c3 = 2.30259  # log(10)
		c4 = 0.434294 # 1./log(10)
		c5 = 1.59603 # log(2)*log(10) = log(2)+log(10)
		c6 = 1.15129 # log(10)/2.
		# changed the following to simplify, see orig function commented out below:
		# delta*(-c6*a-c6*b-c6)   to   -c6*delta*(a+b+1.)
		# (a+b+1.)   to   (IdxHig + 1.)
		
		multiplier = ( (1./(IdxHig+1.)) * -exp(-c1*b) * (cosh(c5*a*delta)-sinh(c5*a*delta)))
		
		lowIntegral = (
			engs[i]**(b+1)
			* ( ((engs[i]/ebk)**(-c2/delta) +1.)**(-c3*a*delta))
			* ((((engs[i]/ebk)**(-c4/delta)) + ((engs[i]/ebk)**(c4/delta)))**(c3*a*delta))
			* float(hyp2f1(-c3*a*delta, -c6*delta*(IdxHig+1.), -c6*delta*(IdxHig+1.)+1., -(engs[i]/ebk)**(-c2/delta)))
			* (cosh((c4 * log(engs[i]/ebk))/delta)**(-c3*a*delta))
			* (((1./cosh((c4*(log(1/ebk)+c1))/delta)) * cosh((c4*log(engs[i]/ebk))/delta))**(c3*a*delta))
			)


		highIntegral = (
			engs[i+1]**(b+1)
			* ( ((engs[i+1]/ebk)**(-c2/delta) +1.)**(-c3*a*delta))
			* ((((engs[i+1]/ebk)**(-c4/delta)) + ((engs[i+1]/ebk)**(c4/delta)))**(c3*a*delta))
			* float(hyp2f1(-c3*a*delta, -c6*delta*(IdxHig+1.), -c6*delta*(IdxHig+1.)+1., -(engs[i+1]/ebk)**(-c2/delta)))
			* (cosh((c4 * log(engs[i+1]/ebk))/delta)**(-c3*a*delta))
			* (((1./cosh((c4*(log(1/ebk)+c1))/delta)) * cosh((c4*log(engs[i+1]/ebk))/delta))**(c3*a*delta))
			)


		val = multiplier * (lowIntegral - highIntegral)
		flux[i] = val


sbplInfo = ("alpha NONE -1.0 -2.0 -1.8 0.8 1.0 0.01",
			"beta NONE -2.0 -5.0 -4.8 -1.0 0.0 0.01",
			"ebreak keV 300. 10. 30. 6000. 7000. 3.0",)
			#"delta NONE 0.3 0.0 0.0 10.0 10.0 0.01",)
AllModels.addPyMod(sbpl, sbplInfo, 'add')
