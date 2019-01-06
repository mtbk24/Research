from __future__ import division
import numpy as np


def log_margin_of_error(value, moe_lower, moe_upper):
    '''
    log_margin_of_error(value, moe_lower, moe_upper)
    
    This function takes unlogged values, lower errors, and upper errors
    and finds the log10 of each. Finding the log10(value) is easy, but 
    the errors are more confusing. You CAN NOT take the log10 of a margin
    of error. You can only take the log10 of the value and the upper and 
    lower confidence intervals. This function does that and returns the 
    correct log10 margins of error. 

    PARAMETERS:
    ----------
    value:  int, float, list, array, or pandas dataframe. 
            The value(s) to be logged with log10.
    moe_lower: int, float, list, array, or pandas dataframe. 
            The lower margin of error. ** must be unlogged **
    moe_upper: int, float, list, array, or pandas dataframe. 
            The upper margin of error. ** must be unlogged **
                
    RETURNS:
    --------
    Returns a list with 3 values:
        logged (base 10) value(s), 
        lower margins of error, 
        upper margins of error.
        
    Values will be returned in the format they were passed to the function. 
    '''
    #import numpy as np
    value = np.asarray(value)
    moe_lower = np.asarray(moe_lower)
    moe_upper = np.asarray(moe_upper)    
    
    # UNLOGGED CONFIDENCE INTERVALS, LOWER AND UPPER.
    CI_L = value - moe_lower
    CI_U = value + moe_upper

    # COMPUTE LOGGED MARGIN OF ERRORS
    loggedmoeL = np.log10( value ) - np.log10( CI_L )
    loggedmoeU = np.log10( CI_U ) - np.log10( value )
    
    return np.log10(value), loggedmoeL, loggedmoeU
    
