'''
File: feedback_linearization.py
Description: implementation of feedback linearization for outer loop / position control
'''

import numpy as np

def feedback_linearization(x_ref,xdot_ref,xddot_ref,y_ref,ydot_ref,yddot_ref,z_ref,zdot_ref,
                           zddot_ref,psi_ref,states,drone_parameters):
    m = drone_parameters.m
    g = drone_parameters.g
    poles_x = drone_parameters.poles_x
    poles_y = drone_parameters.poles_y
    poles_z = drone_parameters.poles_z

    # States needed
    u = states[0]
    v = states[1]
    w = states[2]
    x = states[6]
    y = states[7]
    z = states[8]
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

    # Compute error
    ex = x_ref - x
    ex_dot = xdot_ref - x_dot
    ey = y_ref - y
    ey_dot = ydot_ref - y_dot
    ez = z_ref - z
    ez_dot = zdot_ref - z_dot

    # Compute feedback constants
    kx1 = (poles_x[0] - (poles_x[0] + poles_x[1]) / 2) ** 2 - (poles_x[0] + poles_x[1]) ** 2 / 4
    kx2 = poles_x[0] + poles_x[1]
    kx1 = kx1.real
    kx2 = kx2.real

    ky1 = (poles_y[0] - (poles_y[0] + poles_y[1]) / 2) ** 2 - (poles_y[0] + poles_y[1]) ** 2 / 4
    ky2 = poles_y[0] + poles_y[1]
    ky1 = ky1.real
    ky2 = ky2.real

    kz1 = (poles_z[0] - (poles_z[0] + poles_z[1]) / 2) ** 2 - (poles_z[0] + poles_z[1]) ** 2 / 4
    kz2 = poles_z[0] + poles_z[1]
    kz1 = kz1.real
    kz2 = kz2.real

    # Compute the values vx, vy, vz for the position controller
    ux = kx1 * ex + kx2 * ex_dot
    uy = ky1 * ey + ky2 * ey_dot
    uz = kz1 * ez + kz2 * ez_dot

    vx = xddot_ref - ux
    vy = yddot_ref - uy
    vz = zddot_ref - uz

    # Compute phi, theta, U1
    a = vx / (vz + g)
    b = vy / (vz + g)
    c = np.cos(psi_ref)
    d = np.sin(psi_ref)
    tan_theta = a * c + b * d
    theta_ref = np.arctan(tan_theta)

    # if Psi_ref >= 0:
    #     Psi_ref_singularity = Psi_ref - np.floor(abs(Psi_ref) / (2 * np.pi)) * 2 * np.pi
    # else:
    #     Psi_ref_singularity = Psi_ref + np.floor(abs(Psi_ref) / (2 * np.pi)) * 2 * np.pi
    #
    # if ((np.abs(Psi_ref_singularity) < np.pi / 4 or np.abs(Psi_ref_singularity) > 7 * np.pi / 4) or (
    #         np.abs(Psi_ref_singularity) > 3 * np.pi / 4 and np.abs(Psi_ref_singularity) < 5 * np.pi / 4)):
    #     tan_phi = np.cos(Theta_ref) * (np.tan(Theta_ref) * d - b) / c
    # else:
    #     tan_phi = np.cos(Theta_ref) * (a - np.tan(Theta_ref) * c) / d

    # Avoid singularity
    if np.cos(psi_ref) == 0:
        tan_phi = (a-tan_theta*c)*np.cos(theta_ref) / d
    else:
        tan_phi = (tan_theta*d-b)*np.cos(theta_ref) / c
    phi_ref = np.arctan(tan_phi)
    U1 = (vz + g) * m / (np.cos(phi_ref) * np.cos(theta_ref))

    return phi_ref,theta_ref,U1