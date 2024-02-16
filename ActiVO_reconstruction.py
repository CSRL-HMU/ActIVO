import numpy as np
from CSRL_orientation import *

#given 
#pcf_0
#pcf_1
#Qc_0
#Qc_1
#pc_0
#pc_1


pcf_0 = np.zeros((3, 1000))
pcf_1 = np.zeros((3, 1000))

Qc_0 = np.zeros((4, 1000))
Qc_1 = np.zeros((4, 1000))

pc_0 = np.zeros((3, 1000))
pc_1 = np.zeros((3, 1000))

pf_hat = np.zeros((3, 1000))


N = len(pcf_0[1, :])
dt = 1/N
t = np.array(range(1, N)) * dt

Sigma_c = np.diag((0.01, 0.01, 1))

W = np.zeros((6, 6))
W[0:3, 0:3] = np.linalg.inv(Sigma_c)
W[3:6, 3:6] = np.linalg.inv(Sigma_c)

A = np.zeros((6, 3))
b = np.zeros((6, 1))

for i in range(N):

    Rc_0 = quat2rot(Qc_0[:,i])
    Rc_1 = quat2rot(Qc_1[:,i])

    A[0:3, 0:3] = Rc_0.T
    A[3:6, 0:3] = Rc_1.T

    b[0:3, 0] = pc_0[:,i] + Rc_0 @ pcf_0[:,i]
    b[3:6, 0] = pc_1[:,i] + Rc_1 @ pcf_1[:,i]

    
     
    pf_hat[0:3, i] = (np.linalg.inv(A.T @ W @ A ) @ A.T @ W @ b).T
                      