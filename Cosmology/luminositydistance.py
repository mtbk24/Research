from numpy import sqrt
from scipy import integrate

def LumDist(redshift, CosmoConstants=None):
    '''
    CalcLuminosityDistance(redshift, CosmoConstants=None):
    Given a redshift, returns Luminosity Distance in cm.
    
    To use your own Cosmology Parameters, use 
    CosmoConstants = [H_knot, omega_m]  # [Hubble, Matter Den.]
    
    
    The default Cosmology Constants are:
    H_knot      = 67.8     +- 0.09  km s^-1 Mpc^-1    Hubble Constant.
    omega_m     = 0.308    +- 0.012                   Matter Density.
    omega_l     = 1.0 - omega_m                       Energy Density
    **  Currently using the Planck 2015 results:
    **  https://arxiv.org/pdf/1502.01589.pdf
    
    c           = 2.99792458E+05     Speed of Light in (km/s)
    Mpctocm		= 3.08567758E24		 Megaparsecs to cm conversion
    keVtoerg	= 1.60217657E-09     keV to erg conversion
    MeVtoerg	= 1.60217657E-06     MeV to erg conversion



    # Outdated Cosmology Parameters That Might have been used in analysis.
    # MAY HAVE USED THESE FOR RMFIT.
    omega_m = 0.315
    omega_l = 0.685
    H_knot = 67.3
    
    # THESE MIGHT HAVE BEEN USED FOR XSPEC AND BXA.
    H_knot      = 70.0
    omega_l     = 0.73
    omega_m     = 1.0 - omega_l
    RMFIT and XSPEC/BXA results may have been from different constants.
    
    
    2nd GBM Spectral Catalog states:  
    ' All current Cosmological parameters are obtained from the 
      Planck Collaboration: Ade et al. 2013 '
    
    
    COSMOLOGY CALCULATORS:
    http://www.astro.ucla.edu/~wright/CosmoCalc.html , 
    https://ned.ipac.caltech.edu/help/cosmology_calc.html
    
    '''

    if CosmoConstants is not None:
        H_knot  = CosmoConstants[0]
        omega_m = CosmoConstants[1]
        omega_l = 1.0 - omega_m
    else:
        H_knot      = 67.8              # Hubble Constant.
        omega_m     = 0.308             # Matter Density.
        omega_l     = 1.0 - omega_m
    
    z           = redshift
    #
    c			= 2.99792458e5		 # SPEED OF LIGHT (km/s)
    Mpctocm		= 3.08567758E24		 # Megaparsecs to cm
    keVtoerg	= 1.60217657E-9
    MeVtoerg	= 1.60217657E-6
    

    # Luminosity Distance Calculation:
    def Aint(z):
        eqn = (1./(sqrt(((1.+z)*(1.+z)*(1. + omega_m * z))-(z*(2.+ z)*omega_l ))))
        return eqn
    AA      = integrate.quad(Aint, 0.0, z)
    DL_Mpc  = (c * (1. + z) / H_knot) * AA[0]
    DL_cm   = DL_Mpc * Mpctocm
    return DL_cm






