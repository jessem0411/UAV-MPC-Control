'''
File: ekf_filter.py
Description: simulate IMU measurements by adding some biases and noise
'''

import numpy as np
from sympy import *
from sympy.abc import phi,theta,psi,u,v,w
from scipy.linalg import block_diag


def ekf_filter(estimated_states,covariance_matrix,ang_vel_noisy,acc_noisy,pos_noisy,vel_noisy,angle_noisy,drone_parameters,lambda_g,lambda_a,jacobian_f_x,jacobian_f_u):
    delta_t = drone_parameters.sampling_time
    # Initialize Q and R matrices for EKF
    var_a = 4.8 * 10 ** (-2)
    var_g = 4.8 * 10 ** (-6)
    var_xa = 4 * 10 ** (-14)
    var_xg = var_xa
    var_pos = 10 ** (-3)
    var_vel = 10 ** (-3)
    var_angle = 10 ** (-4)
    Q = block_diag(var_a*np.identity(3),var_g*np.identity(3),var_xa*np.identity(3),var_xg*np.identity(3))
    R = block_diag(var_pos*np.identity(3),var_vel*np.identity(3),var_angle*np.identity(3))

    # Current state estimate: u v w p q r x y z phi theta psi a_x a_y a_z bias_a bias_g
    temp = estimated_states[:,0]
    vel = temp[0:3]
    ang_vel = temp[3:6]
    pos = temp[6:9]
    orientation = temp[9:12]

    acc = temp[12:15]
    bias_a = temp[15:18]
    bias_g = temp[18:21]
    bgx, bgy, bgz, bax, bay, baz = symbols('bgx bgy bgz bax bay baz')
    gyrox, gyroy, gyroz, accx, accy, accz = symbols('gyrox gyroy gyroz accx accy accz')

    # EKF Prediction Equation
    # Propagate dynamics
    phi_num = orientation[0]
    theta_num = orientation[1]
    psi_num = orientation[2]
    T_matrix = np.array([[1, np.sin(phi_num) * np.tan(theta_num), np.cos(phi_num) * np.tan(theta_num)],
                         [0, np.cos(phi_num), -np.sin(phi_num)],
                         [0, np.sin(phi_num) / np.cos(theta_num), np.cos(phi_num) / np.cos(theta_num)]])
    ang_vel_pred = ang_vel_noisy - bias_g

    orientation_pred = orientation + np.matmul(T_matrix,ang_vel_pred) * delta_t
    bias_g_pred = (1-lambda_g*delta_t)*bias_g
    R_x = np.array([[1, 0, 0], [0, np.cos(phi_num), -np.sin(phi_num)], [0, np.sin(phi_num), np.cos(phi_num)]])
    R_y = np.array([[np.cos(theta_num), 0, np.sin(theta_num)], [0, 1, 0], [-np.sin(theta_num), 0, np.cos(theta_num)]])
    R_z = np.array([[np.cos(psi_num), -np.sin(psi_num), 0], [np.sin(psi_num), np.cos(psi_num), 0], [0, 0, 1]])
    R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))
    acc_pred = (acc_noisy-bias_a) + np.matmul(np.linalg.inv(R_matrix),np.array([0,0,9.81]))
    vel_pred = vel + acc_pred * delta_t
    pos_pred = pos + np.matmul(R_matrix,vel_pred * delta_t) + np.matmul(R_matrix,acc_pred / 2 * delta_t ** 2 )
    bias_a_pred = (1-lambda_a*delta_t)*bias_a
    prediction_states = np.concatenate((vel_pred,ang_vel_pred,pos_pred,orientation_pred,acc_pred,bias_a_pred,bias_g_pred),axis=0)

    # Compute Jacobian of f with respect to x and u
    jacobian_f_x = jacobian_f_x((vel[0],vel[1],vel[2],phi_num,theta_num,psi_num,bias_g[0],bias_g[1],bias_g[2],
                                     bias_a[0],bias_a[1],bias_a[2],ang_vel_noisy[0],ang_vel_noisy[1],ang_vel_noisy[2],
                                     acc_noisy[0],acc_noisy[1],acc_noisy[2]))
    jacobian_f_u = jacobian_f_u((phi_num,theta_num,psi_num))
    predicted_covariance_matrix = jacobian_f_x @ covariance_matrix @ np.transpose(jacobian_f_x) +\
                                  jacobian_f_u @ Q @ np.transpose(jacobian_f_u)

    # EKF Measurement Equation
    # y = [x y z u v w phi theta psi]
    H = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0,0,0,0,0]])
    y = np.concatenate((pos_noisy,vel_noisy,angle_noisy),axis=0)
    innovation = y - H @ prediction_states
    S = H @ predicted_covariance_matrix @ np.transpose(H) + R
    K = predicted_covariance_matrix @ np.transpose(H) @ np.linalg.inv(S)
    estimated_states = prediction_states + K @ innovation
    covariance_matrix = (np.identity(21)-K@H)@predicted_covariance_matrix
    estimated_states = np.reshape(estimated_states,(21,-1))

    return estimated_states,covariance_matrix