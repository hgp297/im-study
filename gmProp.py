############################################################################
#             	Intensity measure collector

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	August 2021

# Description: 	Script collects IMs from ground motions including:
# 
# Magnitude
# Rjb

# Open issues: 	(1) 

############################################################################

import pandas as pd

############################################################################
# manual testing
############################################################################

if __name__ == '__main__':
	dataPath = './imStudyData_manualbandpass.csv'

	workData = pd.read_csv(dataPath, sep = ',')
	workData['GMFile'] = workData['GMFile'].astype(str)
	workData.GMFile = workData.GMFile.apply(str).str.strip()

	databasePath = './combinedSearch.csv'
	summary = pd.read_csv(databasePath, skiprows=32, nrows=133)
	summary[' Horizontal-1 Acc. Filename'] = summary[' Horizontal-1 Acc. Filename'].str.replace('.AT2', '')
	
	shortData = pd.DataFrame()
	shortData['Magnitude'] = summary[' Magnitude']
	shortData['Rjb'] = summary[' Rjb (km)']
	shortData['GMFile'] = summary[' Horizontal-1 Acc. Filename']
	shortData['Duration595'] = summary[' 5-95% Duration (sec)']
	shortData['Arias'] = summary[' Arias Intensity (m/sec)']
	shortData.GMFile = shortData.GMFile.apply(str).str.strip()
	
	cols = ['GMFile']
	
	test = workData.join(shortData.set_index(cols), on=cols)