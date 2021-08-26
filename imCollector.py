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
import numpy.fft as npf
import numpy as np
import matplotlib.pyplot as plt
import math

############################################################################
# manual testing
############################################################################

if __name__ == '__main__':
	filePath = './el_centro/RSN179_IMPVALL.H_H-E04140.v3'
	Vt = pd.read_csv(filePath, sep = '\s+', header = None)
	Vt = Vt.to_numpy().flatten()
	Vt = Vt[~np.isnan(Vt)]

	dt = 0.005
	npts = 7818

	timeSeries = np.linspace(0,dt*npts, num=npts)

	Tm = 1

	TLower = 0.2*Tm
	TUpper = 3*Tm
	
	wUpper = 1/TLower
	wLower = 1/TUpper

	wSample = 1/dt
	wRange = [wLower, wUpper]

	b,a = scp.butter(N=4, Wn=wRange, fs=wSample, btype='bandpass', output='ba')
	VFiltered = scp.lfilter(b, a, Vt)

	maxPeaks, maxPeakProps = scp.find_peaks(Vt, height=0)
	maxHeights = maxPeakProps['peak_heights']
	maxLocs = maxHeights.argsort()[-3:][::-1]
	VsMax = maxHeights[maxLocs]

	minPeaks, minPeakProps = scp.find_peaks(-Vt, height=0)
	minHeights = minPeakProps['peak_heights']
	minLocs = minHeights.argsort()[-3:][::-1]
	VsMin = minHeights[minLocs]
	
	FIV = max(np.sum(VsMax), np.sum(VsMin))

	plt.close('all')
	
	fig = plt.figure()
	plt.plot(timeSeries, Vt)
	plt.plot(timeSeries, VFiltered)
	plt.title('Velocity plot')
	plt.ylabel('Velocity (cm/s)')
	plt.xlabel('Time (s)')
	plt.ylim(-50, 50)
	plt.grid(True)

	delT = 0.5*Tm

	windowSize = int(round(delT/dt))

	indexer = np.arange(windowSize)[None,:] + np.arange(npts-windowSize)[:,None]

	Vsquare = np.square(VFiltered)
	integrand = Vsquare[indexer]
	IPTry = np.array([it.simps(row, dx = dt) for row in integrand])
	
	intPower = max(1/delT*IPTry)

	ts2 = np.linspace(0,dt*npts, num=len(IPTry))
	fig = plt.figure()
	plt.plot(ts2, IPTry)
	plt.title('IP plot')
	plt.ylabel('IP (cm2/s2)')
	plt.xlabel('Time (s)')
	plt.grid(True)


############################################################################
# utility functions
############################################################################

def nextpow2(x):
	return 2**(math.ceil(math.log(x, 2)))

# bandpass filter as defined by Zengin
def manual_bandpass(vel, dt, Tm, order=4):

	tUpper = 3.0*Tm
	tLower = 0.2*Tm

	flow = 1/tUpper
	fhigh = 1/tLower

	N = nextpow2(len(vel))

	Y = npf.fft(vel, n=N)
	df = 1/(N*dt)

	# lowpass
	freq = np.arange(0, (N/2)*df, df)
	ff = freq/fhigh

	H = np.zeros(N)

	for i in range(1,len(ff)):
		H[i] = 1/math.sqrt((1+ff[i]**(2*order)))
		Y[i] = Y[i]*H[i]
		Y[len(Y)-i] = Y[len(Y)-i] * H[i]

	# highpass
	freq = np.arange(0, (N/2)*df, df)
	ff = freq/flow

	H = np.zeros(N)

	for i in range(1,len(ff)):
		H[i] = math.sqrt((ff[i]**(2*order))) / math.sqrt((1+ff[i]**(2*order)))
		Y[i] = Y[i]*H[i]
		Y[len(Y)-i] = Y[len(Y)-i] * H[i]

	Y1 = npf.ifft(Y, n=N)
	VFiltered = Y1.real

	return VFiltered


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
	# in Hz (1/s)
	low = 1/tUpper
	high = 1/tLower
	b, a = scp.butter(order, [low, high], fs = fs, btype='bandpass')
	return b, a

# get intensity measures from a predominant period, GM name, scale
def getIMs(row, T, GMDir, gmData):

	# get dt and npts of the GM file
	dt = getDT(row['GMFile'], gmData)
	npts = getNPTS(row['GMFile'], gmData)

	# calculate sampling frequency (Hz, 1/s)
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
	# VFiltered = butter_bandpass_filter(Vt, row[T], fs)

	# Zengin's bandpass
	VFiltered = manual_bandpass(Vt, dt, row[T], order=4)

	# Zengin and Abrahamson (2020)
	delT = 0.5*row[T]
	windowSize = int(round(delT/dt))

	# get sliding window of values between t and t+deltaT
	indexer = np.arange(windowSize)[None,:] + np.arange(npts-windowSize)[:,None]
	VSquare = np.square(VFiltered)

	# use trapezoidal method to integrate each window according to Eq. 3
	IPt = np.array([it.trapz(row, dx = dt) for row in VSquare[indexer]])

	# calculate IP
	instantPower = max(1/delT*IPt)

	# instantPower = max(1/delT * np.convolve(np.square(VFiltered),
	# 	np.ones(windowSize)*dt, 'valid'))

	# Heresi and Miranda (2020)
	# apply lowpass filter on acceleration
	AFiltered = butter_lowpass_filter(At, fs)
	aT = 0.7*row[T]

	# collect the velocity_s series for each window according to Eq. 9
	windowSize2=  int(round(aT/dt))
	g = 386.4
	indexer2 = np.arange(windowSize2)[None,:] + np.arange(npts-windowSize2)[:,None]
	ASlice = AFiltered[indexer2]*g
	Vs = np.array([it.trapz(row, dx = dt) for row in ASlice])

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

	return(instantPower, FIV3, PGV, PGA)

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

# newDat = main()