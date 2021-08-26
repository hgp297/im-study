############################################################################
#             	Intensity measure collector

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	June 2021

# Description: 	Script adds intensity motions by re-entering the search csv and 
# manipulating ground motion records

# Open issues: 	(1) 

############################################################################

import pandas as pd
import numpy as np
import re

pd.options.mode.chained_assignment = None  # default='warn'


def getST(row, Ttype, summary, unscaledSpectra):
	
	GMFile = row['GMFile']
	scaleFactor = row['GMScale']
	Tquery = row[Ttype]

	rsn 				= re.search('(\d+)', GMFile).group(1)
	gmUnscaledName		= 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
	gmSpectrum			= unscaledSpectra[['Period (sec)', gmUnscaledName]]
	gmSpectrum.columns	= ['Period', 'Sa']

	SaQueryUnscaled 	= np.interp(Tquery, gmSpectrum.Period, gmSpectrum.Sa)
	SaQuery 			= scaleFactor*SaQueryUnscaled
	return(SaQuery)

def getSaAvgTm(row, summary, unscaledSpectra):
	
	GMFile = row['GMFile']
	scaleFactor = row['GMScale']
	Tm = row['Tm']

	rsn 				= re.search('(\d+)', GMFile).group(1)
	gmUnscaledName		= 'RSN-' + str(rsn) + ' Horizontal-1 pSa (g)'
	gmSpectrum			= unscaledSpectra[['Period (sec)', gmUnscaledName]]
	gmSpectrum.columns	= ['Period', 'Sa']

	gmSpectrum['ScaledSa'] = gmSpectrum['Sa']*scaleFactor

	# calculate desired target spectrum average (0.2*Tm, 1.5*Tm)
	tLower 	= 0.2*Tm
	tUpper	= 1.5*Tm

	# geometric mean from Eads et al. (2015)
	spectrumRange = gmSpectrum[gmSpectrum['Period'].between(tLower, tUpper)]['ScaledSa']
	spectrumAverage = spectrumRange.prod()**(1/spectrumRange.size)

	return(spectrumAverage)


if __name__ == '__main__':
	isolDat = pd.read_csv('./random600.csv')
	resultsCSV = 'combinedSearch.csv'
	gmDir = './'

	summaryStart = 32 
	nSummary = 133
	unscaledStart = 290
	nUnscaled = 111

	# load in sections of the sheet
	summary = pd.read_csv(gmDir+resultsCSV, 
		skiprows=summaryStart, nrows=nSummary)
	unscaledSpectra = pd.read_csv(gmDir+resultsCSV, 
		skiprows=unscaledStart, nrows=nUnscaled)

	# get Sa(T) values at Tfb
	isolDat['GMSTfb'] = isolDat.apply(getST, axis = 1, 
		args = ['Tfb', summary, unscaledSpectra])

	# get Sa_avg centered around Tm
	isolDat['GmSavgTm'] = isolDat.apply(getSaAvgTm, axis = 1,
		args = [summary, unscaledSpectra])
	
	isolDat.to_csv('./imStudyData.csv', index = False)
