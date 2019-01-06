In the majority of my research, I need to log (base 10) my data and the margins of error on that data. 

You CAN NOT log the margins of error. 
You MUST find the confidence intervals (lower and upper) and then log those and then take the difference between the logged x and y values and their respective logged upper and lower confidence intervals. 

To see how this is done, see loggeddata.py

Most of our data sets have asymmetric errors and logging the data has become a pain. This function is useful anytime the logged data is needed. 

testing_loggingdata.ipynb
 - Uses this function to log the data. Also shows some plots. 
 
asymmetric_errorbars.ipynb
 - Introduces some of the pains of using plt.errorbar with asymmetric error bars. In certain circumstances, plt.errorbar will plot symmetric errorbars, only using the lower error value, which is not desired. 