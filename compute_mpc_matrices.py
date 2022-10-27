'''
File: compute_mpc_matrices.py
Description: compute MPC matrices needed to solve for optimal control
'''

import numpy as np

def compute_mpc_matrices(Ad,Bd,Cd,Dd,max_horizon,drone_parameters,y_min,y_max):
    # ABCD matrices for augmented system
    Atilde = np.concatenate((np.concatenate((Ad,Bd),axis=1),
             np.concatenate((np.zeros((np.size(Bd,1),np.size(Ad,1))),np.identity(np.size(Bd,1))),axis=1)),axis=0)
    Btilde = np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
    Ctilde = np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
    Dtilde = Dd

    # Weight matrices
    Q = drone_parameters.Q
    S = drone_parameters.S
    R = drone_parameters.R

    CQC = np.matmul(np.matmul(np.transpose(Ctilde),Q),Ctilde)
    CSC = np.matmul(np.matmul(np.transpose(Ctilde),S),Ctilde)
    QC = np.matmul(Q,Ctilde)
    SC = np.matmul(S, Ctilde)

    temp = np.kron(np.eye(max_horizon-1,dtype=int),CQC)
    temp2 = np.kron(np.eye(max_horizon-1,dtype=int),QC)
    Qdbar = np.concatenate((np.concatenate((temp,np.zeros((np.size(temp,0),np.size(CSC,1)))),axis=1),
            np.concatenate((np.zeros((np.size(CSC,0),np.size(temp,1))),CSC),axis=1)),axis=0)
    Tdbar = np.concatenate((np.concatenate((temp2,np.zeros((np.size(temp2,0),np.size(SC,1)))),axis=1),
            np.concatenate((np.zeros((np.size(SC,0),np.size(temp2,1))),SC),axis=1)),axis=0)
    Rdbar = np.kron(np.eye(max_horizon,dtype=int),R)

    Cdbar = np.zeros((np.size(Atilde,0)*max_horizon,np.size(Btilde,1)*max_horizon))
    Adhat = np.zeros((np.size(Atilde,0)*max_horizon,np.size(Atilde,1)))

    Ctildestar = np.concatenate((np.zeros((3,6)),np.identity(3)),axis=1)
    Cgtildestar = np.kron(np.eye(max_horizon,dtype=int),Ctildestar)
    y_min_global = np.zeros((np.size(y_min,0)*max_horizon,np.size(y_min,1)))
    y_max_global = np.zeros((np.size(y_max, 0) * max_horizon, np.size(y_max, 1)))

    for i in range(0,max_horizon):
        for j in range(0,max_horizon):
            if j <= i:
                Cdbar[i*Btilde.shape[0]:(i+1)*Btilde.shape[0],j*Btilde.shape[1]:(j+1)*Btilde.shape[1]] = np.matmul(np.linalg.matrix_power(Atilde,i-j),Btilde)
        Adhat[i*Atilde.shape[0]:(i+1)*Atilde.shape[0],:] = np.linalg.matrix_power(Atilde,i+1)
        y_min_global[i*y_min.shape[0]:(i+1)*y_min.shape[0],:] = y_min
        y_max_global[i*y_max.shape[0]:(i + 1) * y_max.shape[0], :] = y_max

    Hdbar = np.matmul(np.matmul(np.transpose(Cdbar),Qdbar),Cdbar) + Rdbar
    Fdbart = np.concatenate((np.matmul(np.matmul(np.transpose(Adhat),Qdbar),Cdbar),np.matmul(-Tdbar,Cdbar)),axis=0)

    return Hdbar,Fdbart,Cdbar,Adhat,Cgtildestar,y_min_global,y_max_global