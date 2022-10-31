'''
File: simulate_imu_measurements.py
Description: simulate IMU measurements by adding some biases and noise
'''

import numpy as np

def simulate_gps_measurements(states):
    pos = np.reshape(states[6:9],(3,1))
    vel = np.reshape(states[0:3],(3,1))
    angle = np.reshape(states[9:12],(3,1))

    # Variance of noise
    var_pos = 10 ** (-3)
    var_vel = 10 ** (-3)
    var_angle = 10 ** (-4)

    y_p = pos + np.sqrt(var_pos)*np.random.randn(3,1)
    y_v = vel + np.sqrt(var_vel)*np.random.randn(3,1)
    y_angle = angle + np.sqrt(var_angle)*np.random.randn(3,1)

    return y_p[:,0],y_v[:,0],y_angle[:,0]