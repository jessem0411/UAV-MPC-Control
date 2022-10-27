'''
File: uav_dynamics.py
Description: propagate the state of the UAV according to the dynamics (Runge-Kutta)
'''

import numpy as np

def uav_dynamics(states,omega,T,Mx,My,Mz,drone_parameters):
    # Set constants
    Ixx = drone_parameters.Ixx
    Iyy = drone_parameters.Iyy
    Izz = drone_parameters.Izz
    m = drone_parameters.m
    g = drone_parameters.g
    Jtp = drone_parameters.Jtp
    sampling_time = drone_parameters.sampling_time

    # States definition
    current_states = states
    new_states = current_states
    u = current_states[0]
    v = current_states[1]
    w = current_states[2]
    p = current_states[3]
    q = current_states[4]
    r = current_states[5]
    x = current_states[6]
    y = current_states[7]
    z = current_states[8]
    phi = current_states[9]
    theta = current_states[10]
    psi = current_states[11]
    sub_loop = drone_parameters.sub_loop
    states_animation = np.zeros((sub_loop, 6))
    input_animation = np.zeros((sub_loop, 4))

    # Drag force:
    drag_switch = drone_parameters.drag_switch
    C_D_u = drone_parameters.C_D_u
    C_D_v = drone_parameters.C_D_v
    C_D_w = drone_parameters.C_D_w
    A_u = drone_parameters.A_u
    A_v = drone_parameters.A_v
    A_w = drone_parameters.A_w
    rho = drone_parameters.rho

    # Runge-Kutta implementation
    for k in range(0,4):
        if drag_switch == 1:
            Fd_u = 0.5 * C_D_u * rho * u ** 2 * A_u
            Fd_v = 0.5 * C_D_v * rho * v ** 2 * A_v
            Fd_w = 0.5 * C_D_w * rho * w ** 2 * A_w
        else:
            Fd_u = 0
            Fd_v = 0
            Fd_w = 0

        # Compute slopes
        u_dot = (v * r - w * q) + g * np.sin(theta) - Fd_u / m
        v_dot = (w * p - u * r) - g * np.cos(theta) * np.sin(phi) - Fd_v / m
        w_dot = (u * q - v * p) - g * np.cos(theta) * np.cos(phi) + T / m - Fd_w / m
        p_dot = q * r * (Iyy - Izz) / Ixx + Jtp / Ixx * q * omega + Mx / Ixx
        q_dot = p * r * (Izz - Ixx) / Iyy - Jtp / Iyy * p * omega + My / Iyy
        r_dot = p * q * (Ixx - Iyy) / Izz + Mz / Izz

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

        # Save slopes:
        if k == 0:
            slopes_k1 = np.array([u_dot,v_dot,w_dot,p_dot,q_dot,r_dot,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot])
            next_temp_states = current_states + slopes_k1 * sampling_time/2
        elif k == 1:
            slopes_k2 = np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot])
            next_temp_states = current_states + slopes_k2 * sampling_time / 2
        elif k == 2:
            slopes_k3 = np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot])
            next_temp_states = current_states + slopes_k3 * sampling_time
        else:
            slopes_k4 = np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot])
            next_temp_states = current_states + 1/6*(slopes_k1+2*slopes_k2+2*slopes_k3+slopes_k4) * sampling_time

        u = next_temp_states[0]
        v = next_temp_states[1]
        w = next_temp_states[2]
        p = next_temp_states[3]
        q = next_temp_states[4]
        r = next_temp_states[5]
        x = next_temp_states[6]
        y = next_temp_states[7]
        z = next_temp_states[8]
        phi = next_temp_states[9]
        theta = next_temp_states[10]
        psi = next_temp_states[11]

    new_states = next_temp_states
    for k in range(0, sub_loop):
        x_or = current_states[6]
        y_or = current_states[7]
        z_or = current_states[8]
        phi_or = current_states[9]
        theta_or = current_states[10]
        psi_or = current_states[11]
        states_animation[k, 0] = x_or + (x - x_or) / sampling_time * (k / (sub_loop - 1)) * sampling_time
        states_animation[k, 1] = y_or + (y - y_or) / sampling_time * (k / (sub_loop - 1)) * sampling_time
        states_animation[k, 2] = z_or + (z - z_or) / sampling_time * (k / (sub_loop - 1)) * sampling_time
        states_animation[k, 3] = phi_or + (phi - phi_or) / sampling_time * (k / (sub_loop - 1)) * sampling_time
        states_animation[k, 4] = theta_or + (theta - theta_or) / sampling_time * (k / (sub_loop - 1)) * sampling_time
        states_animation[k, 5] = psi_or + (psi - psi_or) / sampling_time * (k / (sub_loop - 1)) * sampling_time
        input_animation[k,0] = T
        input_animation[k, 1] = Mx
        input_animation[k, 2] = My
        input_animation[k, 3] = Mz

    return new_states,states_animation,input_animation