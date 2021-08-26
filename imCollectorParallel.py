############################################################################
#             	Intensity measure collector

# Created by: 	Huy Pham
# 				University of California, Berkeley

# Date created:	June 2021

# Description: 	Script collects IMs from ground motions including:
# 
# filtered incremental velocity (FIV3) (Davalos and Miranda, 2019)
# instantaneous power IM as defined by Zengin and Abrahamson (2020)
# PGA
# PGV

# Open issues: 	(1) 

############################################################################

import pandas as pd
import scipy.signal as scp
import scipy.integrate as it
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

############################################################################
# manual testing
############################################################################

# if __name__ == '__main__':
# 	filePath = './el_centro/RSN179_IMPVALL.H_H-E04140.v3'
# 	Vt = pd.read_csv(filePath, sep = '\s+', header = None)
# 	Vt = Vt.to_numpy().flatten()
# 	Vt = Vt[~np.isnan(Vt)]

# 	dt = 0.005
# 	npts = 7818

# 	timeSeries = np.linspace(0,dt*npts, num=npts)

# 	Tm = 1

# 	TLower = 0.2*Tm
# 	TUpper = 3*Tm
	
# 	wUpper = 2*np.pi/TLower
# 	wLower = 2*np.pi/TUpper

# 	wSample = 1/dt
# 	wRange = [wLower, wUpper]

# 	b,a = scp.butter(N=4, Wn=wRange, fs=wSample, btype='bandpass', output='ba')
# 	# sos = scp.butter(N=4, Wn=wRange, fs=wSample, btype='bandpass', output='sos')
# 	# VFiltered = scp.sosfilt(sos, Vt)
# 	VFiltered = scp.lfilter(b, a, Vt)

# 	maxPeaks, maxPeakProps = scp.find_peaks(Vt, height=0)
# 	maxHeights = maxPeakProps['peak_heights']
# 	maxLocs = maxHeights.argsort()[-3:][::-1]
# 	VsMax = maxHeights[maxLocs]

# 	minPeaks, minPeakProps = scp.find_peaks(-Vt, height=0)
# 	minHeights = minPeakProps['peak_heights']
# 	minLocs = minHeights.argsort()[-3:][::-1]
# 	VsMin = minHeights[minLocs]	

# 	plt.close('all')
	
# 	fig = plt.figure()
# 	plt.plot(timeSeries, Vt)
# 	plt.plot(timeSeries, VFiltered)
# 	plt.title('Velocity plot')
# 	plt.ylabel('Velocity (in/s)')
# 	plt.xlabel('Time (s)')
# 	plt.grid(True)

# 	delT = 0.5*Tm

# 	windowSize = int(round(delT/dt))

# 	indexer = np.arange(windowSize)[None,:] + np.arange(npts-windowSize)[:,None]

# 	Vsquare = np.square(VFiltered)
# 	integrand2 = Vsquare[indexer]
# 	IPTry = np.array([it.simps(row, dx = dt) for row in integrand2])

# 	integrand = np.convolve(Vsquare, np.ones(windowSize,dtype=int)*dt, 'valid')

# 	intPower = max(1/delT*np.convolve(np.square(VFiltered), 
# 		np.ones(windowSize,dtype=int)*dt, 'valid'))
	
# 	intPower2 = max(1/delT*IPTry)


############################################################################
# utility functions
############################################################################

# filter signal with Butterworth bandpass filter
# convert num/den parameters into signal
def butter_lowpass_filter(acc, fs, order=2):
	high = 1
	b, a = scp.butter(order, high, fs = fs, btype = 'lowpass')
	y = scp.lfilter(b, a, acc)
	return y

# filter signal with Butterworth bandpass filter
# convert num/den parameters into signal
def butter_bandpass_filter(vel, T, fs, order=4):
	tLower = 0.2*T
	tUpper = 3.0*T
	b, a = butter_bandpass(tLower, tUpper, fs, order=order)
	y = scp.lfilter(b, a, vel)
	return y

# implement Butterworth bandpass filter
def butter_bandpass(tLower, tUpper, fs, order=4):
	pi = 3.14159
	low = 2*pi/tUpper
	high = 2*pi/tLower
	b, a = scp.butter(order, [low, high], fs = fs, btype='bandpass')
	return b, a

# get intensity measures from a predominant period, GM name, scale
def getIMs(row, T, GMDir, gmData):

	# get dt and npts of the GM file
	dt = getDT(row['GMFile'], gmData)
	npts = getNPTS(row['GMFile'], gmData)

	# calculate sampling frequency
	fs = 1/dt

	# get corresponding velocity series
	velPath = GMDir + row['GMFile'] + '.VT2'
	Vt = pd.read_csv(velPath, sep = '\s+', header = None)
	Vt = Vt.to_numpy()
	# remove any NaN's and scale to match GM used in analysis
	Vt = Vt[~np.isnan(Vt)]*row['GMScale']

	# get corresponding acceleration series
	accPath = GMDir + row['GMFile'] + '.g3'
	At = pd.read_csv(accPath, sep = '\s+', header = None)
	At = At.to_numpy().flatten()
	# remove any NaN's and scale to match GM used in analysis
	At = At[~np.isnan(At)]*row['GMScale']

	# get PGV (in/s) and PGA (g)
	PGV = max(Vt)
	PGA = max(At)

	# apply filter around [0.2T, 3.0T]
	VFiltered = butter_bandpass_filter(Vt, row[T], fs)

	# Zengin and Abrahamson (2020)
	delT = 0.5*row[T]
	windowSize = int(round(delT/dt))

	# get sliding window of values between t and t+deltaT
	indexer = np.arange(windowSize)[None,:] + np.arange(npts-windowSize)[:,None]
	VSquare = np.square(VFiltered)

	# use simpson's method to integrate each window according to Eq. 3
	VsqParam = [[series, dt] for series in VSquare[indexer]]

	# Heresi and Miranda (2020)
	# apply lowpass filter on acceleration
	AFiltered = butter_lowpass_filter(At, fs)
	aT = 0.7*row[T]

	# collect the velocity_s series for each window according to Eq. 9
	windowSize2=  int(round(aT/dt))
	g = 386.4
	indexer2 = np.arange(windowSize2)[None,:] + np.arange(npts-windowSize2)[:,None]
	ASlice = AFiltered[indexer2]*g
	AParam = [[series, dt] for series in ASlice]

	# use parallel computation
	with mp.Pool(processes=4) as pool:
		IPtHolder = pool.map(trapint, VsqParam)
		VsHolder = pool.map(trapint, AParam)

	# calculate IP
	IPt = np.array(IPtHolder)
	instantPower = max(1/delT*IPt)

	# calculate FIV3
	Vs = np.array(VsHolder)

	# collect largest 3 local minima/maxima
	maxPeaks, maxPeakProps = scp.find_peaks(Vs, height=0)
	maxHeights = maxPeakProps['peak_heights']
	maxLocs = maxHeights.argsort()[-3:][::-1]
	VsMax = maxHeights[maxLocs]

	minPeaks, minPeakProps = scp.find_peaks(-Vs, height=0)
	minHeights = minPeakProps['peak_heights']
	minLocs = minHeights.argsort()[-3:][::-1]
	VsMin = minHeights[minLocs]

	# calculate FIV3
	FIV3 = max(np.sum(VsMax), np.sum(VsMin))
	
	# close mp pools
	pool.close()
	pool.join()

	return(instantPower, FIV3, PGV, PGA)

# integrate in parallel function
def trapint(args):
	y = args[0]
	dx = args[1]
	return(it.trapz(y, dx=dx))

# get sampling dt
def getDT(GMFile, gmData):
	return(float(gmData.loc[gmData['GM_Name'] == GMFile, 'DT']))

# get sampling dt
def getNPTS(GMFile, gmData):
	return(int(gmData.loc[gmData['GM_Name'] == GMFile, 'NPTS']))


############################################################################
# main call
############################################################################

def main():
	gmDir = './PEERNGARecords_Unscaled/'
	gmData = pd.read_csv(gmDir+'gmInfo.csv')
	isolDat = pd.read_csv('./imStudyData.csv')

	isolDat['IPTm'], isolDat['FIV3Tm'], isolDat['PGV'], isolDat['PGA'] = zip(*isolDat.apply(getIMs,
		axis = 1, args = ['Tm', gmDir, gmData]))

	return(isolDat)

if __name__=="__main__":
	mp.freeze_support()
	newDat = main()
