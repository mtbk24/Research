# This file holds the results from the bxa fitting process.  It holds the best fit parameters based on the Multinest algorithm use of the Maximum Likelihood.  Holds best fit parameters, errors, and statistics.
from __future__ import division
import os
import numpy
from numpy import exp, log
import pandas as pd
from collections import OrderedDict
from astropy.io import fits as pyfits
import json
import matplotlib.pyplot as plt


from PyMultinest_plot_marginals import PlotMarginal
from PyMultinest_analyse import Analyzer




modName	  = 'grbm+blackb'
version	  = '-01-'
det		  = 'G'
burst	  = 'bn080916009'


def Run_Marginal_Plots(modName, version, det, burst):
	detector	 = ('GBMwLAT' if 'L' in det else 'GBM')

	data_dir	 = "/Users/KimiZ/GRBs2/analysis/LAT/%s/integrated/"%burst
	mod_dir	     = "/Users/KimiZ/GRBs2/analysis/LAT/%s/BXA/%s/%s/"%(burst, detector, modName)
	pyx_dir      = "/Users/KimiZ/GRBs2/analysis/LAT/%s/PYXSPEC/%s/%s/"%(burst, detector, modName)
	base_dir	 = "/Users/KimiZ/GRBs2/analysis/pyxspec_programs/Analysis/"

	Prefix		 = '%s_%s_%s_'%(modName, version, det)
	Prefix_ext   = os.path.join(mod_dir, '%s_%s_%s_'%(modName, version, det))
	
	pyx_file     = "xspec_fitresults_%s.fit"%Prefix



	def get_ParNames(model):
		'''
		Given a model name, it will return the parameter names of the model.
		See file 'ParameterNames.json in the $PYX/Analysis/ directory.
		'
		'''
		fname   = "/Users/KimiZ/GRBs2/analysis/pyxspec_programs/Analysis/ParameterNames.json"
		f	   = json.load(open(fname, 'r'), encoding='utf-8')
		names   = f[model] # DICTIONARY OF PARAMETERS
		names   = [str(i) for i in names]
		return names

	Pars	 = get_ParNames(modName)
	nPars	= len(Pars)


	fN = os.path.join(pyx_dir, pyx_file)

	f = pyfits.open(fN)

	PYX_bestfit = OrderedDict()
	for i,name in enumerate(f[2].data.columns.names):
		if 'PARAM' in name:
			value = f[2].data[name][0][0]
			if 'norm' in Pars[i]:
				PYX_bestfit["%s__%i"%(Pars[i],i+1)]   = numpy.log10(value)
			else:
				PYX_bestfit["%s__%i"%(Pars[i],i+1)]   = value
		else:
			pass


	analyser = Analyzer(nPars, Prefix_ext)




	def marginal_plots(analyzer, d=None, XLims='5sigma'):
		"""
		Create marginal plots
	
		* analyzer: A instance of pymultinest.Analyzer
		* d: if more than 20 dimensions, by default only 1d marginal distributions
		   are plotted. set 
		   d=2 if you want to force a 2d matrix plot
		   set 
		   d=1 if you want to save the top row of the 2d plot
		   (probability dist of each parameter) into its own file.
		* XLims: '1sigma', '2sigma', '3sigma' or '5sigma'
					'5sigma' is the default.
	
		"""
		prefix	  = analyzer.outputfiles_basename
		n_params	= analyzer.n_params
		parameters  = json.load(file(prefix + 'params.json'))
		a		   = analyzer
		s		   = a.get_stats()
	
		p		   = PlotMarginal(a)  # moved a copy of file to this direc.
	
		#p = pymultinest.PlotMarginal(a) # call to /Users/KimiZ/BXA-master/bxa/xspec/plot.py
	
		values	 = a.get_equal_weighted_posterior()
		assert n_params == len(s['marginals'])
		modes	  = s['modes']
		
		
		
		# IF YOU DON'T WANT THE BOTTOM HALF OF PLOTS, USE d = anything but 2
		# IF YOU WANT THE 2D MARGINALS, USE d=2 or leave it out.
		dim2 = ((1 if n_params > 20 else 2) if d is None else d) == 2
	
		if dim2:
			plt.figure(figsize=(5*n_params, 5*n_params))
			for i in range(n_params):
				plt.subplot(n_params, n_params, i + 1)
				plt.xlabel(parameters[i])
	
				m = s['marginals'][i]
				plt.xlim(m[XLims])
	
				oldax = plt.gca()
				x,w,patches = oldax.hist(values[:,i], bins=20, edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
				oldax.set_ylim(0, x.max())
	
				newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
				p.plot_marginal(i, ls='-', color='blue', linewidth=3)
				newax.set_ylim(0, 1)
	
				ylim = newax.get_ylim()
				y = ylim[0] + 0.05*(ylim[1] - ylim[0])
				center = m['median']
				low1, high1 = m['1sigma']
				#print center, low1, high1
				newax.errorbar(x=center, y=y,
					xerr=numpy.transpose([[center - low1, high1 - center]]), 
					color='blue', linewidth=2, marker='s')
				
				
				bestFit = s['modes'][0]['maximum'][i]
				newax.vlines(bestFit, 0, 1, linestyle='--', color='black', lw=1, label="%f"%bestFit)
				newax.vlines(center, 0, 1, linestyle='-.', color='blue', lw=1, label="%f"%center)
	
				oldax.set_yticks([])
				#newax.set_yticks([])  I didn't comment this out, they did
				newax.set_ylabel("Probability")
				ylim = oldax.get_ylim()
				newax.set_xlim(m[XLims])
				oldax.set_xlim(m[XLims])
				newax.legend(loc='best')
	
				for j in range(i):
					plt.subplot(n_params, n_params, n_params * (j + 1) + i + 1)
					p.plot_conditional(i, j, bins=20, cmap = plt.cm.gray_r)
					for m in modes:
						plt.errorbar(x=m['mean'][i], y=m['mean'][j], xerr=m['sigma'][i], yerr=m['sigma'][j])
					plt.xlabel(parameters[i])
					plt.ylabel(parameters[j])
					plt.xlim(s['marginals'][i][XLims])
					plt.ylim(s['marginals'][j][XLims])
					#plt.savefig('cond_%s_%s.pdf' % (params[i], params[j]), bbox_tight=True)
					#plt.close()
			plt.tight_layout() # kz keep axes from overlapping
			#plt.show()
			print(prefix)
			plt.savefig(prefix + 'marg2d_%s.pdf'%XLims)
			
		else:
			from matplotlib.backends.backend_pdf import PdfPages
			print '1dimensional only. Set the D environment variable D=2 to force'
			print '2d marginal plots.'
			pp = PdfPages(prefix + 'marg1d_%s.pdf'%XLims)
	
			for i in range(n_params):
				plt.figure(figsize=(5, 5))
				plt.xlabel(parameters[i])
	
				m = s['marginals'][i]
				plt.xlim(m[XLims])
	
				oldax = plt.gca()
				x,w,patches = oldax.hist(values[:,i], bins=20, edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
				oldax.set_ylim(0, x.max())
	
				newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
				p.plot_marginal(i, ls='-', color='blue', linewidth=3)
				newax.set_ylim(0, 1)
	
				ylim = newax.get_ylim()
				y = ylim[0] + 0.05*(ylim[1] - ylim[0])
				center = m['median']
				low1, high1 = m['1sigma']
				print center, low1, high1
				newax.errorbar(x=center, y=y,
					xerr=numpy.transpose([[center - low1, high1 - center]]), 
					color='blue', linewidth=2, marker='s')
							
				bestFit = s['modes'][0]['maximum'][i]
				newax.vlines(bestFit, 0, 1, linestyle='--', color='black', lw=1, label="%f"%bestFit)
				newax.vlines(center, 0, 1, linestyle='-.', color='blue', lw=1, label="%f"%center)
	
				oldax.set_yticks([])
				newax.set_ylabel("Probability")
				ylim = oldax.get_ylim()
				newax.set_xlim(m[XLims])
				oldax.set_xlim(m[XLims])
				plt.legend(loc='best')
				#plt.show()
				plt.savefig(pp, format='pdf', bbox_inches='tight')
			pp.close()
	
	marginal_plots(analyser, d=2, XLims='5sigma')
	marginal_plots(analyser, d=1, XLims='5sigma')
	
	marginal_plots(analyser, d=2, XLims='3sigma')
	marginal_plots(analyser, d=1, XLims='3sigma')


	#os.system('open %s'%mod_dir)
	

	
	

	def marginal_plots_1donly(analyzer, XLims='5sigma'):
		"""
		Create marginal plots

		* analyzer: A instance of pymultinest.Analyzer
		* d: if more than 20 dimensions, by default only 1d marginal distributions
		   are plotted. set d=2 if you want to force a 2d matrix plot

		"""
		prefix      = analyzer.outputfiles_basename
		n_params    = analyzer.n_params
		parameters  = json.load(file(prefix + 'params.json'))
		a           = analyzer
		s           = a.get_stats()

		p           = PlotMarginal(a)  # moved a copy of file to this direc.

		#p = pymultinest.PlotMarginal(a) # call to /Users/KimiZ/BXA-master/bxa/xspec/plot.py

		values     = a.get_equal_weighted_posterior()
		assert n_params == len(s['marginals'])
		modes      = s['modes']
		
		plt.figure(figsize=(5*n_params, 5*n_params))
		from matplotlib.backends.backend_pdf import PdfPages
		print '1dimensional only. Set the D environment variable D=2 to force'
		print '2d marginal plots.'
		pp = PdfPages(prefix + 'marg1d_%s_wPYX.pdf'%XLims)

		for i in range(n_params):
			plt.figure(figsize=(7,7))
			plt.xlabel(parameters[i])

			m = s['marginals'][i]
			plt.xlim(m[XLims])

			oldax = plt.gca()
			x,w,patches = oldax.hist(values[:,i], bins=20, edgecolor='grey', color='grey', histtype='stepfilled', alpha=0.2)
			oldax.set_ylim(0, x.max())

			newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
			p.plot_marginal(i, ls='-', color='blue', linewidth=1)
			newax.set_ylim(0, 1)

			ylim = newax.get_ylim()
			y = ylim[0] + 0.05*(ylim[1] - ylim[0])
			center = m['median']
			low1, high1 = m['1sigma']
			print center, low1, high1
			newax.errorbar(x=center, y=y,
				xerr=numpy.transpose([[center - low1, high1 - center]]), 
				color='blue', linewidth=2, marker='s')
						
			bestFit = s['modes'][0]['maximum'][i]
			newax.vlines(bestFit, 0, 1, linestyle='--', color='black', lw=1, label="BXA best: %f"%bestFit)
			
			PYXbestFit = PYX_bestfit["%s__%i"%(parameters[i], i+1)]
			newax.vlines(PYXbestFit, 0, 1, linestyle='--', color='brown', lw=1, label="PYX best: %f"%PYXbestFit)

			newax.vlines(center, 0, 1, linestyle='-.', color='blue', lw=1, label="BXA median: %f"%center)

			oldax.set_yticks([])
			newax.set_ylabel("Probability")
			ylim = oldax.get_ylim()
			newax.set_xlim(m[XLims])
			oldax.set_xlim(m[XLims])
			plt.legend(loc='best', fontsize=8)
			#plt.show()
			plt.savefig(pp, format='pdf', bbox_inches='tight')
		pp.close()

	marginal_plots_1donly(analyser, XLims='5sigma')
	marginal_plots_1donly(analyser, XLims='3sigma')


