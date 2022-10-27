'''
File: cont_to_discrete.py
Description: continuous system to discretized state space
'''

import numpy as np
def cont_to_discrete(states,omega,drone_parameters):
    Ixx = drone_parameters.Ixx
    Iyy = drone_parameters.Iyy
    Izz = drone_parameters.Izz
    Jtp = drone_parameters.Jtp
    sampling_time = drone_parameters.sampling_time

    # States needed
    u = states[0]
    v = states[1]
    w = states[2]
    p = states[3]
    q = states[4]
    r = states[5]
    phi = states[9]
    theta = states[10]
    psi = states[11]

    # Compute xdot ydot zdot
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))
    vel_body = np.array([[u], [v], [w]])
    vel_fixed = np.matmul(R_matrix, vel_body)
    x_dot = vel_fixed[0]
    y_dot = vel_fixed[1]
    z_dot = vel_fixed[2]
    x_dot = x_dot[0]
    y_dot = y_dot[0]
    z_dot = z_dot[0]

    # Compute phi_dot theta_dot psi_dot
    T_matrix = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                         [0, np.cos(phi), -np.sin(phi)],
                         [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
    rot_vel_body = np.array([[p], [q], [r]])
    rot_vel_fixed = np.matmul(T_matrix, rot_vel_body)
    phi_dot = rot_vel_fixed[0]
    theta_dot = rot_vel_fixed[1]
    psi_dot = rot_vel_fixed[2]
    phi_dot = phi_dot[0]
    theta_dot = theta_dot[0]
    psi_dot = psi_dot[0]

    # Create ABCD matrices for continuous system and AdBdCdDd for discretized system
    A = np.array([[0,1,0,0,0,0],
                  [0,0,0,Jtp*omega/Ixx,0,(Iyy-Izz)*theta_dot/Ixx],
                  [0,0,0,1,0,0],
                  [0,-Jtp*omega/Iyy,0,0,0,(Izz-Ixx)*phi_dot/Iyy],
                  [0,0,0,0,0,1],
                  [0,(Ixx-Iyy)*theta_dot/(2*Izz),0,(Ixx-Iyy)*phi_dot/(2*Izz),0,0]])
    B = np.array([[0,0,0],
                  [1/Ixx,0,0],
                  [0,0,0],
                  [0,1/Iyy,0],
                  [0,0,0],
                  [0,0,1/Izz]])
    C = np.array([[1,0,0,0,0,0],
                  [0,0,1,0,0,0],
                  [0,0,0,0,1,0]])
    D = np.zeros((3,3))
    Ad = np.identity(np.size(A, 1)) + sampling_time * A
    Bd = sampling_time * B
    Cd = C
    Dd = D

    return Ad, Bd, Cd, Dd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot