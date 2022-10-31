'''
File: simulate_imu_measurements.py
Description: simulate IMU measurements by using the true state and adding some biases and noise
'''

import numpy as np

def simulate_imu_measurements(states,acc,bias_g,bias_a,lambda_g,lambda_a,drone_parameters):
    delta_t = drone_parameters.sampling_time
    ang_vel = np.reshape(states[3:6],(3,1))
    acc = np.reshape(acc,(3,1))
    phi = states[9]
    theta = states[10]
    psi = states[11]
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

    # Variance of noise and bias
    var_a = 4.8*10 ** (-2)
    var_g = 4.8 * 10 ** (-6)
    var_xa = 4 * 10 ** (-14)
    var_xg = var_xa

    bias_a = np.reshape(bias_a,(3,1))
    bias_g = np.reshape(bias_g, (3, 1))
    bias_a = (1-lambda_a*delta_t)*bias_a + np.sqrt(var_xa)*np.random.randn(3,1)
    bias_g = (1-lambda_g*delta_t)*bias_g + np.sqrt(var_xg)*np.random.randn(3,1)
    gravity = np.array([0,0,9.81])
    gravity = np.reshape(gravity,(3,1))

    # Model the noisy measurements

    y_g = ang_vel + bias_g + np.sqrt(var_g)*np.random.randn(3,1)
    y_a = acc + bias_a + np.sqrt(var_a) * np.random.randn(3, 1) - np.matmul(R_matrix,gravity)

    return y_g[:,0],y_a[:,0],bias_a,bias_g