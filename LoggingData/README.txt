In the majority of my research, I need to log (base 10) my data and the margins of error on that data. In addition, most of my data sets have asymmetric errors and logging this data properly has become a pain. I wrote a function to return properly logged values and their margins of error in the file: loggeddata.py

You CAN NOT log margins of error -- you MUST log confidence intervals!!

Other files:
testing_loggingdata.ipynb
 - Uses this function to log the data. Also shows some plots. 
 
asymmetric_errorbars.ipynb
 - Introduces some of the pains of using plt.errorbar with asymmetric error bars. In certain circumstances, plt.errorbar will plot symmetric errorbars, only using the lower error value, which is not desired. 