from ctypes import sizeof
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math as m
from scipy.interpolate import interp1d
import pandas as pd
import os
import scipy.io


def modify_the_signal(signal, t):
  T = 1
  modified_signal = np.zeros(len(t))
  for i in range(len(t)):
    modified_signal[i] = (signal[-1] - signal[0]) / T * t[i] + signal[0]

  return modified_signal


def new_signal(interpolated_signal2, modify_the_signal1, modify_the_signal2,
               t):
  NEWSIGNAL = np.ones((len(t)))
  for i in range(len(t)):

    NEWSIGNAL[i] = interpolated_signal2[i] - (modify_the_signal2[i] -
                                              modify_the_signal1[i])
  return NEWSIGNAL


pf_hat1 = (pd.read_excel('pf_hat1.xlsx').to_numpy()).T
pf_hat2 = (pd.read_excel('pf_hat2.xlsx').to_numpy()).T

t_ref = (pd.read_excel('t_ref.xlsx').to_numpy()).T
t_2 = (pd.read_excel('t2.xlsx').to_numpy()).T

#common time vector
common_t_exp1 = np.linspace(0, 1, 1000)  # common time

# # Interpolate both signals onto the common time vector
# interpolated_signal1 = np.interp(common_t, t1, signal1)
# interpolated_signal2 = np.interp(common_t, t2, signal2)

# Interpolate both signals onto the common time vector
# print(t_ref)
# print(pf_hat1)
fx_1 = interp1d(np.linspace(0, 1, len(pf_hat1[:, 0])), pf_hat1[:, 0])
fy_1 = interp1d(np.linspace(0, 1, len(pf_hat1[:, 0])), pf_hat1[:, 1])
fz_1 = interp1d(np.linspace(0, 1, len(pf_hat1[:, 0])), pf_hat1[:, 2])

fx_2 = interp1d(np.linspace(0, 1, len(pf_hat2[:, 0])), pf_hat2[:, 0])
fy_2 = interp1d(np.linspace(0, 1, len(pf_hat2[:, 0])), pf_hat2[:, 1])
fz_2 = interp1d(np.linspace(0, 1, len(pf_hat2[:, 0])), pf_hat2[:, 2])

signal_1_X = modify_the_signal(fx_1(common_t_exp1), common_t_exp1)
signal_1_Y = modify_the_signal(fy_1(common_t_exp1), common_t_exp1)
signal_1_Z = modify_the_signal(fz_1(common_t_exp1), common_t_exp1)

signal_2_X = modify_the_signal(fx_2(common_t_exp1), common_t_exp1)
signal_2_Y = modify_the_signal(fy_2(common_t_exp1), common_t_exp1)
signal_2_Z = modify_the_signal(fz_2(common_t_exp1), common_t_exp1)

signal_2_new_X = new_signal(fx_2(common_t_exp1), signal_1_X, signal_2_X,
                            common_t_exp1)
signal_2_new_Y = new_signal(fy_2(common_t_exp1), signal_1_Y, signal_2_Y,
                            common_t_exp1)
signal_2_new_Z = new_signal(fz_2(common_t_exp1), signal_1_Z, signal_2_Z,
                            common_t_exp1)

signal_1_new_X = fx_1(common_t_exp1)
signal_1_new_Y = fy_1(common_t_exp1)
signal_1_new_Z = fz_1(common_t_exp1)

#pf_hat2 = modify_the_signal(interpolated_pf_hat2,t2 ,common_

# signal_2_new_X = new_signal(pf_hat2[:, 0], pf_hat1[:, 0], common_t_exp1)
# signal_2_new_Y = new_signal(pf_hat2[:, 1], pf_hat1[:, 1], common_t_exp1)
# signal_2_new_Z = new_signal(pf_hat2[:, 2], pf_hat1[:, 2], common_t_exp1)

# signal_2_new_X = new_signal(pf_hat2[:, 0], pf_hat1[:, 0], common_t_exp1)
# signal_2_new_Y = new_signal(pf_hat2[:, 1], pf_hat1[:, 1], common_t_exp1)
# signal_2_new_Z = new_signal(pf_hat2[:, 2], pf_hat1[:, 2], common_t_exp1)

signal_1_new = (np.vstack((signal_1_new_X, signal_1_new_Y, signal_1_new_Z))).T
signal_2_new = (np.vstack((signal_2_new_X, signal_2_new_Y, signal_2_new_Z))).T

# signal_1_X = pf_hat1[:, 0]
# signal_1_Y = pf_hat1[:, 1]
# signal_1_Z = pf_hat1[:, 2]
# pf_hat1 = (np.vstack((signal_1_X, signal_1_Y, signal_1_Z))).T

distance, path = fastdtw(signal_1_new, signal_2_new, dist=euclidean)

#distance_3_2, path_3_2 = fastdtw(signal2, signal_3_new)

# Get the alignment path

# Create aligned signals

aligned_signal2 = np.array([signal_2_new[i] for i, j in path])
# print(aligned_signal2)
# print(pf_hat1)
#####################################################
#####_____________________________________##########
# Plot the signals and the alignment path
# plt.figure(figsize=(12, 6))

# plt.subplot(3, 1, 1)
# plt.plot(signal_1_new, label='NEW_pf_hat1')
# #plt.title('pf_hat1')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(signal_2_new, label='Signal 2')
# plt.title('Signal 2')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(aligned_signal1, label='Aligned Signal 1')

# plt.title('Aligned Signals')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()

# plt.tight_layout()
# plt.show()

#####################################################
#####_____________________________________##########
# Resample to a common length (e.g., 500 points)
common_length = 1000
print(np.linspace(0, 1, len(pf_hat1[:, 0])))
print('-------------------------')
print(aligned_signal2[:, 0])
f2X = interp1d(np.linspace(0, 1, len(aligned_signal2[:, 0])),
               aligned_signal2[:, 0],
               kind='linear')
f2Y = interp1d(np.linspace(0, 1, len(aligned_signal2[:, 0])),
               aligned_signal2[:, 1],
               kind='linear')
f2Z = interp1d(np.linspace(0, 1, len(aligned_signal2[:, 0])),
               aligned_signal2[:, 2],
               kind='linear')

f1X = interp1d(np.linspace(0, 1, len(pf_hat1[:, 0])),
               pf_hat1[:, 0],
               kind='linear')

f1Y = interp1d(np.linspace(0, 1, len(pf_hat1[:, 1])),
               pf_hat1[:, 1],
               kind='linear')

f1Z = interp1d(np.linspace(0, 1, len(pf_hat1[:, 2])),
               pf_hat1[:, 2],
               kind='linear')
# f3 = interp1d(np.linspace(0, 1, len(aligned_signal3)),
#               aligned_signal3,
#               kind='linear')

resampled_signal2_X = f2X(np.linspace(0, 1, common_length))
resampled_signal2_Y = f2Y(np.linspace(0, 1, common_length))
resampled_signal2_Z = f2Z(np.linspace(0, 1, common_length))

resampled_signal1_X = f1X(np.linspace(0, 1, common_length))
resampled_signal1_Y = f1Y(np.linspace(0, 1, common_length))
resampled_signal1_Z = f1Z(np.linspace(0, 1, common_length))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(resampled_signal2_X,
        resampled_signal2_Y,
        resampled_signal2_Z,
        label='FINAL_pf_hat2')
ax.plot(resampled_signal1_X,
        resampled_signal1_Y,
        resampled_signal1_Z,
        label='FINAL_pf_hat1(Ref Signal)')

plt.axis('equal')
plt.legend()
plt.title('Final Signals')
plt.show()
