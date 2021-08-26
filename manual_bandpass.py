import pandas as pd
import scipy.signal as scp
import scipy.integrate as it
import numpy.fft as npf
import numpy as np
import matplotlib.pyplot as plt
import math

def nextpow2(x):
	return 2**(math.ceil(math.log(x, 2)))

if __name__ == '__main__':
	filePath = './el_centro/RSN179_IMPVALL.H_H-E04140.v3'
	Vt = pd.read_csv(filePath, sep = '\s+', header = None)
	Vt = Vt.to_numpy().flatten()
	Vt = Vt[~np.isnan(Vt)]

	dt = 0.005
	npts = 7818

	timeSeries = np.linspace(0,dt*npts, num=npts)

	Tm = 1

	tLower = 0.2*Tm
	tUpper = 3*Tm

	flow = 1/tUpper
	fhigh = 1/tLower

	order = 4
	N = nextpow2(len(Vt))

	Y = npf.fft(Vt, n=N)

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

	ts2 = np.linspace(0,dt*N, num=len(VFiltered))

	plt.close('all')
	fig = plt.figure()
	plt.plot(timeSeries, Vt)
	plt.plot(ts2, VFiltered)
	plt.title('Velocity plot')
	plt.ylabel('Velocity (cm/s)')
	plt.xlabel('Time (s)')
	plt.ylim(-50, 50)
	plt.grid(True)

	delT = 0.05*Tm

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


# bandpass filter as defined by Zengin
def manual_bandpass(vel, dt, tUpper, tLower, order=4):
	flow = 1/tUpper
	fhigh = 1/tLower

	N = nextpow2(len(Vt))

	Y = npf.fft(vel, n=N)
	df = 1/(N*dt)

	# lowpass
	freq = np.arange(0, (N/2)*df, df)
	ff = freq/fhigh

	for i in range(1,len(ff)):
		H[i] = 1/math.sqrt((1+ff[i]**(2*order)))
		Y[i] = Y[i]*H[i]
		Y[len(Y)-i+2] = Y[len(Y)-i+2] * H[i]

	# highpass
	freq = np.arange(0, (N/2)*df, df)
	ff = freq/flow

	for i in range(1,len(ff)):
		H[i] = math.sqrt((ff[i]**(2*order))) / math.sqrt((1+ff[i]**(2*order)))
		Y[i] = Y[i]*H[i]
		Y[len(Y)-i+2] = Y[len(Y)-i+2] * H[i]

	Y1 = npf.ifft(Y, n=N)
	Y1 = Y1.real

	return Y1