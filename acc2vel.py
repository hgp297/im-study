############################################################################
#             	Accel2Velocity Converter

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	June 2021

# Description: 	Script converts PEER ground motion files from AT2 format to VT2 
# format using trapezoidal integration

# Open issues: 	(1) 

############################################################################
# import
############################################################################

import pandas as pd
import scipy.integrate as it
import matplotlib.pyplot as plt
import numpy as np
import os
from ReadRecord import ReadRecord

############################################################################
# manual checking
############################################################################

# filePath = './PEERNGARecords_Unscaled/RSN347_COALINGA.H_H-Z09000.g3'

# g = 386.4
# accelSeries = pd.read_csv(filePath, sep = '\s+', header = None)
# accelSeries = accelSeries.to_numpy().flatten()*g

# dt = 0.01
# npts = 6500
# velSeries = it.cumtrapz(accelSeries, dx = dt, initial = 0)

# timeSeries = np.linspace(0,dt*npts, num=npts)

# plt.close('all')

# fig = plt.figure()
# plt.plot(timeSeries, velSeries)
# plt.title('Velocity plot')
# plt.ylabel('Velocity (in/s)')
# plt.xlabel('Time (s)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(timeSeries, accelSeries)
# plt.title('Acceleration plot')
# plt.ylabel('Acceleration (in/s^2)')
# plt.xlabel('Time (s)')
# plt.grid(True)


# truePath = './coalinga/RSN347_COALINGA.H_H-Z09000.g3'
# trueVel = './coalinga/RSN347_COALINGA.H_H-Z09000.v3'
# accelTrueSeries = pd.read_csv(truePath, sep = '\s+', header = None)
# accelTrueSeries = accelTrueSeries.to_numpy().flatten()*g
# velTrueSeries = pd.read_csv(trueVel, sep = '\s+', header = None)
# cmtoin = 0.393701
# velTrueSeries = velTrueSeries.to_numpy().flatten()*cmtoin

# fig = plt.figure()
# plt.plot(timeSeries, velTrueSeries)
# plt.title('True Velocity plot')
# plt.ylabel('Velocity (in/s)')
# plt.xlabel('Time (s)')
# plt.grid(True)

# fig = plt.figure()
# plt.plot(timeSeries, accelTrueSeries)
# plt.title('True Acceleration plot')
# plt.ylabel('Acceleration (in/s^2)')
# plt.xlabel('Time (s)')
# plt.grid(True)

#############################################################################
# conversion tool
#############################################################################

def accelConvert(GMFile):
	GMDir = './PEERNGARecords_Unscaled/'

	inFile 	= GMDir + GMFile + '.AT2'
	outFile = GMDir + GMFile + '.g3'		# set variable holding new filename 

	# call procedure to convert the ground-motion file
	dt, nPts = ReadRecord(inFile, outFile)

	# read in acceleration series and flatten from 5xn to 1xn
	g = 386.4
	accel = pd.read_csv(outFile, sep = '\s+', header = None)
	accel = accel.to_numpy().flatten() 	# read left to right
	accel = accel[~np.isnan(accel)]*g

	# use trapezoidal integration from scipy (cumu sum method)
	vel = it.cumtrapz(accel, dx = dt, initial = 0)

	return(vel, dt, nPts)

#############################################################################
# main script to convert all file in folder
#############################################################################

def main():

	# enter directory
	GMDir = './PEERNGARecords_Unscaled/'
	folder = os.listdir(GMDir)

	metaDf = None

	for item in folder:

		# grab name of ground motion used
		if item.endswith('.g3'):
			GMFile = item.replace('.g3', '')
		else:
			continue

		# convert accel to velocity (in/s)
		velArray, dt, nPts = accelConvert(GMFile)

		# write velocity array to file
		velName = GMDir + GMFile + '.VT2'
		vFile = open(velName, "w")
		
		np.savetxt(vFile, velArray)

		vFile.close()

		# save metadata
		GMData = dict()

		GMData['GM_Name'] = GMFile
		GMData['DT'] = dt
		GMData['NPTS'] = nPts

		dataRow = pd.DataFrame.from_dict(GMData, 'index').transpose()

		if metaDf is None:
			metaDf = pd.DataFrame(columns=['GM_Name', 'DT', 'NPTS'])

		metaDf = pd.concat([dataRow, metaDf], sort=False)

	metaDf.to_csv(GMDir + 'gmInfo.csv', index=False)


#############################################################################

main()