import matplotlib.pyplot as plt
import numpy as np

from gaussian import *

currents1 = np.array([1, 1.3, 1.66, 1.88, 2, 2.27, 6.86, 8.05, 12.33, 13.76, 16.05, 17.3])
volts1 = np.array([0.9, 1.14, 1.46, 1.63, 1.71, 1.94, 5.42, 6.28, 9.3, 10.35, 11.85, 12.65])

T2 = 23 #Celsius
currents2 = np.array([2.95, 4.28, 5.94, 8.02, 13.84, 10.85]) #mA
volts2 = np.array([3.11, 4.55, 6.15, 8.04, 12.6, 10.17]) #v
params = np.polyfit(currents2, volts2, 1)

T3 = 48 # EMF = 1.94mV
currents3 = np.array([2.94, 7.3, 10.45, 13.05, 5.15])
volts3 = np.array([3.43, 7.74, 10.5, 12.64, 5.6])

T4 = 91 # EMF = 3.72
currents4 = np.array([19.85, 15.5, 11.36, 8.37, 7, 3.19, ])
volts4 = np.array([9.45, 8.2, 6.61, 5.22, 4.6, 2.34, ])

T5 = 70 # EMF = 2.85
currents5 = np.array([1.02, 5.24, 10, 15.5, 7, 12.07])
volts5 = np.array([1.2, 5.5, 9, 12.25, 6.8, 10.6])

T6 = 111 # EMF = 4.55
currents6 = np.array([1, 7.07, 11.33, 19.88, 15.05, 4.04])
volts6 = np.array([0.66, 3.62, 4.7, 7.1, 5.9, 2.25])

T7 = 141 # EMF = 5.75
currents7 = np.array([1.2, 19.85, 15.35, 11.1, 7.77, 3.2, 1])
volts7 = np.array([0.4, 4.1, 3.3, 2.7, 1.88, 0.86, 0.28])

T8 = 50 # EMF = 2.02
currents8 = np.array([1.13, 12.7, 9.1, 6, 3.6, 6.25])
volts8 = np.array([1.59, 12.35, 9.35, 6.6, 4.3, 6.9])

T9 = 80 # EMF = 3.26
currents9 = np.array([0.75, 1.5, 5.05, 8, 9.4, 13, 18.7, 14.7])
volts9 = np.array([0.85, 1.7, 4.85, 6.88, 7.5, 9.7, 12.46, 10.5])

T10 = 130 # EMF = 5.33
currents10 = np.array([0.45,3.6, 7.97,11.5,15.18,17.54,19.4])
volts10 = np.array([0.18,1.4,2.9,3.7,4.5,4.8,4.9])

# T11 between 100 and 130C not measured
T11 = 100 # EMF = 4.1
currents11 = np.array([19.88,16.3,13,10,6.97,3,0.37])
volts11 = np.array([7.8,7.1,6.1,5.3,4.1,2.1,0.29])

plt.scatter(currents2, volts2, label=f"T = {T2}")
plt.scatter(currents3, volts3, label=f"T = {T3}")
plt.scatter(currents4, volts4, label=f"T = {T4}")
plt.scatter(currents5, volts5, label=f"T = {T5}")
plt.scatter(currents6, volts6, label=f"T = {T6}")
plt.scatter(currents7, volts7, label=f"T = {T7}")
plt.scatter(currents8, volts8, label=f"T = {T8}")
plt.scatter(currents9, volts9, label=f"T = {T9}")
plt.scatter(currents10, volts10, label=f"T = {T10}")
plt.scatter(currents11, volts11, label=f"T = {T11}")

def linear(x, m):
    return m * x
x_range = np.linspace(min(currents11), max(currents11), 1000)
params, cov_mat = curve_fit(linear, currents11, volts11)
m = params[0]
plt.plot(x_range, linear(x_range, m), label=f"Fit T={T11}C: V={m:.2f}I")



plt.xlabel("I[mA]")
plt.ylabel("V[V]")
plt.legend()
plt.show()
