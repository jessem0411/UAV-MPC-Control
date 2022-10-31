'''
File: get_jacobian.py
Description: compute Jacobian matrices for EKF
'''

import numpy as np
from sympy import sin, cos, tan, Matrix,symbols
from sympy.abc import phi,theta,psi,p,q,r,u,v,w,x,y,z
from sympy import lambdify

def get_jacobian(drone_parameters,lambda_g,lambda_a):
    delta_t = drone_parameters.sampling_time
    gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z = symbols('gyrox gyroy gyroz accx accy accz')
    bias_g_x,bias_g_y,bias_g_z,bias_a_x,bias_a_y,bias_a_z = symbols('bgx bgy bgz bax bay baz')
    ax,ay,az = symbols('ax ay az')
    bgux,bguy,bguz,baux,bauy,bauz = symbols('bgux bguy bguz baux bauy bauz')

    R_x = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    R_y = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    R_z = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    R_matrix = R_z*R_y*R_x
    inv_R_matrix = R_matrix.inv()
    T_matrix = Matrix([[1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                         [0, cos(phi), -sin(phi)],
                         [0, sin(phi) / cos(theta), cos(phi) / cos(theta)]])
    orientation = Matrix([phi,theta,psi])
    ang_vel = Matrix([p,q,r])
    gyro = Matrix([gyro_x,gyro_y,gyro_z])
    bias_g = Matrix([bias_g_x,bias_g_y,bias_g_z])
    bias_a = Matrix([bias_a_x, bias_a_y, bias_a_z])
    acc = Matrix([acc_x,acc_y,acc_z])
    gravity = Matrix([0,0,9.81])
    vel = Matrix([u,v,w])
    pos = Matrix([x,y,z])
    acceleration = Matrix([ax,ay,az])
    bias_g_uncertain = Matrix([bgux,bguy,bguz])
    bias_a_uncertain = Matrix([baux, bauy, bauz])

    # Compute propagation function
    f_ang_vel = gyro - bias_g
    f_orientation = orientation + T_matrix*f_ang_vel*delta_t
    f_bias_g = (1-lambda_g*delta_t)*bias_g + bias_g_uncertain

    f_bias_a = (1-lambda_a*delta_t)*bias_a + bias_a_uncertain
    f_acc = acc - bias_a + inv_R_matrix*gravity
    f_vel = vel + f_acc*delta_t
    f_pos = pos + R_matrix * f_vel * delta_t + R_matrix * f_acc * (delta_t ** 2) / 2
    ftotal = Matrix([f_vel,f_ang_vel,f_pos,f_orientation,f_acc,f_bias_a,f_bias_g])
    # Define the state
    X = Matrix([u,v,w,p,q,r,x,y,z,phi,theta,psi,ax,ay,az,bias_a_x,bias_a_y,bias_a_z,bias_g_x,bias_g_y,bias_g_z])

    # Define the input
    U = Matrix([acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,bias_a_uncertain,bias_g_uncertain])

    # Compute Jacobian
    Jx = ftotal.jacobian(X)
    Ju = ftotal.jacobian(U)
    Jx = lambdify([(u, v, w, phi, theta, psi, bias_g_x, bias_g_y, bias_g_z, bias_a_x, bias_a_y,
                                   bias_a_z, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z)], Jx)
    Ju = lambdify([(phi,theta,psi)],Ju)
    return Jx,Ju