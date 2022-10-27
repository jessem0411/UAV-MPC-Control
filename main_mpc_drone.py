'''
File: main_mpc_drone.py
Description: Implementation of MPC with feedback linearization for trajectory tracking of UAV

'''

import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from initialize_parameters import *
from trajectory_generation import *
from feedback_linearization import *
from cont_to_discrete import *
from compute_mpc_matrices import *
from uav_dynamics import *
from qpsolvers import solve_qp

# Initialize parameters
drone_parameters = InitializeParameters()
sampling_time = drone_parameters.sampling_time
num_att_outputs = drone_parameters.num_att_outputs
inner_loop_length = drone_parameters.inner_loop_length
sub_loop = drone_parameters.sub_loop

# Create trajectory
Tf = 100     # final time
time_step = sampling_time * inner_loop_length # time step for trajectory generation
time_trajectory = np.arange(0,Tf + time_step,time_step) # time vector for trajectory generation
time_mpc = np.arange(0,Tf + sampling_time,sampling_time) # time vector for inner loop MPC
time_animation = np.arange(0,Tf + sampling_time/sub_loop,sampling_time/sub_loop)
x_ref,xdot_ref,xddot_ref,y_ref,ydot_ref,yddot_ref,z_ref,zdot_ref,zddot_ref,psi_ref = trajectory_generation(drone_parameters,time_trajectory)

# Create initial state vector
states = np.array([0,0,0,0,0,0,0,-1,0,0,0,psi_ref[0]])     # u v w p q r x y z phi theta psi
states_history = [states]
states_history_animation = [states[6:len(states)]]
angles_ref_history = np.array([[0,0,0]])
velocity_fixed_history = np.array([[0,0,0]])

# Create initial propeller velocity (rad/s)
omega_min = drone_parameters.omega_min
omega_max = drone_parameters.omega_max
omega1 = omega_min
omega2 = omega_min
omega3 = omega_min
omega4 = omega_min
omega = omega1 - omega2 + omega3 - omega4
Ct = drone_parameters.Ct
Cq = drone_parameters.Cq
l = drone_parameters.l

# Compute initial thrust and moments
T = Ct*(omega1**2 + omega2**2 + omega3**2 +omega4**2)
Mx = Ct*l*(omega4**2 - omega2**2)
My = Ct*l*(omega3**2 - omega1**2)
Mz = Cq*(-omega1**2 + omega2**2 - omega3**2 + omega4**2)
input_history = np.array([[T,Mx,My,Mz]])
propeller_history = np.array([[omega1,omega2,omega3,omega4]])
input_history_animation = input_history

# Compute moments constraints
Mx_min = Ct*l*(omega_min**2-omega_max**2)
Mx_max = Ct*l*(omega_max**2-omega_min**2)
My_min = Ct*l*(omega_min**2-omega_max**2)
My_max = Ct*l*(omega_max**2-omega_min**2)
Mz_min = Cq*2*(omega_min**2-omega_max**2)
Mz_max = Cq*2*(omega_max**2-omega_min**2)
y_min = np.array([[Mx_min],[My_min],[Mz_min]])
y_max = np.array([[Mx_max],[My_max],[Mz_max]])

#### Implementation of MPC with feedback linearization ####
for i in range(0,len(time_trajectory)-1):
    # Position control
    phi_ref,theta_ref,T = feedback_linearization(x_ref[i+1],xdot_ref[i+1],xddot_ref[i+1],y_ref[i+1],
                                                 ydot_ref[i+1],yddot_ref[i+1],z_ref[i+1],zdot_ref[i+1],
                                                 zddot_ref[i+1],psi_ref[i+1],states,drone_parameters)
    phi_ref_mpc = np.transpose([phi_ref*np.ones(inner_loop_length+1)])
    theta_ref_mpc = np.transpose([theta_ref*np.ones(inner_loop_length+1)])

    # Check thrust constraints
    T_min = Ct*4* omega_min**2
    T_max = Ct * 4 * omega_max ** 2
    if T < T_min:
        T = T_min
    elif T > T_max:
        T = T_max

    # Make Psi_ref increase continuosly in a linear fashion per outer loop so that control is smooth
    psi_ref_mpc = np.transpose([np.zeros(inner_loop_length + 1)])
    for yaw_step in range(0, inner_loop_length + 1):
        psi_ref_mpc[yaw_step] = psi_ref[i] + (psi_ref[i + 1] - psi_ref[i]) / (sampling_time * inner_loop_length) * sampling_time * yaw_step

    angles_ref = np.concatenate((phi_ref_mpc[1:len(phi_ref_mpc)], theta_ref_mpc[1:len(theta_ref_mpc)],
                                  psi_ref_mpc[1:len(psi_ref_mpc)]),axis=1)
    angles_ref_history = np.concatenate((angles_ref_history, angles_ref), axis=0)

    # Create reference angles for MPC controller
    # refSignal = [phi_ref_0, theta_ref_0, psi_ref_0, phi_ref_1, theta_ref_2, psi_ref_2, ... etc.]
    angles_ref_mpc = np.zeros(len(psi_ref_mpc) * num_att_outputs)
    k = 0
    for i in range(0, len(angles_ref_mpc), num_att_outputs):
        angles_ref_mpc[i] = phi_ref_mpc[k]
        angles_ref_mpc[i + 1] = theta_ref_mpc[k]
        angles_ref_mpc[i + 2] = psi_ref_mpc[k]
        k = k + 1

    # Initialize inner loop controller
    max_horizon = drone_parameters.max_horizon   # maximum horizon period
    k = 0
    for j in range(0,inner_loop_length):
        # Discretize state space
        Ad, Bd, Cd, Dd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = cont_to_discrete(states,omega,drone_parameters)
        x_dot = np.transpose([x_dot])
        y_dot = np.transpose([y_dot])
        z_dot = np.transpose([z_dot])
        velocity_fixed = np.concatenate(([[x_dot], [y_dot], [z_dot]]), axis=1)
        velocity_fixed_history = np.concatenate((velocity_fixed_history, velocity_fixed), axis=0)
        # Create augmented state with [angles; previous moments input]
        x_tilde = np.transpose([np.concatenate(([phi, phi_dot, theta, theta_dot, psi, psi_dot], [Mx, My, Mz]), axis=0)])

        # Extract reference angles for the inner loop
        k = k + num_att_outputs
        current_angles_ref = angles_ref_mpc[k:len(angles_ref_mpc)]
        if j > 0:
            max_horizon = max_horizon-1

        # Compute matrices needed for MPC (Hdoublebar,Fdoublebar,Cdoublebar,Ahat)
        Hdbar,Fdbart,Cdbar,Adhat,Cgtildestar,y_min_global,y_max_global = compute_mpc_matrices(Ad,Bd,Cd,Dd,max_horizon,drone_parameters,y_min,y_max)
        aug_x_and_ref = np.concatenate((x_tilde,np.reshape(current_angles_ref,(-1,1))),axis=0)
        f = np.matmul(np.transpose(Fdbart),aug_x_and_ref)

        CC = np.matmul(Cgtildestar,Cdbar)
        G = np.concatenate((CC,-CC),axis=0)
        CAX = np.matmul(np.matmul(Cgtildestar,Adhat),x_tilde)
        h1 = y_max_global - CAX
        h2 = -y_min_global + CAX
        h = np.concatenate((h1,h2),axis=0)

        deltau = solve_qp(Hdbar,f,G,np.transpose(h)[0],solver="cvxopt")


        # Compute optimal control from MPC
        # deltau = -np.matmul(np.matmul(np.linalg.inv(Hdbar),np.transpose(Fdbart)),aug_x_and_ref)

        # Update control inputs
        Mx = Mx + deltau[0]
        My = My + deltau[1]
        Mz = Mz + deltau[2]
        Mx = Mx.item()
        My = My.item()
        Mz = Mz.item()

        input_history = np.concatenate((input_history, np.array([[T, Mx, My, Mz]])), axis=0)
        # Compute omega (propeller rotational velocity)
        Ustar = np.array([T/Ct,Mx/(Ct*l),My/(Ct*l),Mz/Cq])
        Sstar = np.array([[1,1,1,1],[0,1,0,-1],[-1,0,1,0],[-1,1,-1,1]])
        temp_omegas = np.matmul(np.linalg.inv(Sstar),Ustar)
        if any(o < 0 for o in temp_omegas) == True:
            print("Inadmissible values for control inputs!")
        else:
            omega1 = np.sqrt(temp_omegas[0])
            omega2 = np.sqrt(temp_omegas[1])
            omega3 = np.sqrt(temp_omegas[2])
            omega4 = np.sqrt(temp_omegas[3])

        propeller_history = np.concatenate((propeller_history,np.array([[omega1,omega2,omega3,omega4]])),axis=0)
        omega = omega1-omega2+omega3-omega4

        # Propagate UAV dyanmics
        states,states_animation,input_animation = uav_dynamics(states,omega,T,Mx,My,Mz,drone_parameters)
        states_history = np.concatenate((states_history,[states]),axis=0)
        states_history_animation = np.concatenate((states_history_animation, states_animation), axis=0)
        input_history_animation = np.concatenate((input_history_animation, input_animation), axis=0)

################################ ANIMATION LOOP ###############################
if max(y_ref)>=max(x_ref):
    max_ref=max(y_ref)
else:
    max_ref=max(x_ref)

if min(y_ref)<=min(x_ref):
    min_ref=min(y_ref)
else:
    min_ref=min(x_ref)

states_history_x=states_history_animation[:,0]
states_history_y=states_history_animation[:,1]
states_history_z=states_history_animation[:,2]
states_history_phi=states_history_animation[:,3]
states_history_theta=states_history_animation[:,4]
states_history_psi=states_history_animation[:,5]
input_history_T=input_history_animation[:,0]
input_history_Mx=input_history_animation[:,1]
input_history_My=input_history_animation[:,2]
input_history_Mz=input_history_animation[:,3]
frame_amount=int(len(states_history_x))
drone_length_x=max_ref*0.1 # Length of one half of the UAV in the x-direction (Only for the animation purposes)
drone_length_y=max_ref*0.1 # Length of one half of the UAV in the y-direction (Only for the animation purposes)

def update_plot(num):

    # Rotational matrix that relates u,v,w with x_dot,y_dot,z_dot
    R_x=np.array([[1, 0, 0],[0, np.cos(states_history_phi[num]), -np.sin(states_history_phi[num])],[0, np.sin(states_history_phi[num]), np.cos(states_history_phi[num])]])
    R_y=np.array([[np.cos(states_history_theta[num]),0,np.sin(states_history_theta[num])],[0,1,0],[-np.sin(states_history_theta[num]),0,np.cos(states_history_theta[num])]])
    R_z=np.array([[np.cos(states_history_psi[num]),-np.sin(states_history_psi[num]),0],[np.sin(states_history_psi[num]),np.cos(states_history_psi[num]),0],[0,0,1]])
    R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))

    drone_pos_body_x=np.array([[drone_length_x],[0],[0]])
    drone_pos_inertial_x=np.matmul(R_matrix,drone_pos_body_x)

    drone_pos_body_x_neg=np.array([[-drone_length_x],[0],[0]])
    drone_pos_inertial_x_neg=np.matmul(R_matrix,drone_pos_body_x_neg)

    drone_pos_body_y=np.array([[0],[drone_length_y],[0]])
    drone_pos_inertial_y=np.matmul(R_matrix,drone_pos_body_y)

    drone_pos_body_y_neg=np.array([[0],[-drone_length_y],[0]])
    drone_pos_inertial_y_neg=np.matmul(R_matrix,drone_pos_body_y_neg)

    drone_body_x.set_xdata([states_history_x[num]+drone_pos_inertial_x_neg[0][0],states_history_x[num]+drone_pos_inertial_x[0][0]])
    drone_body_x.set_ydata([states_history_y[num]+drone_pos_inertial_x_neg[1][0],states_history_y[num]+drone_pos_inertial_x[1][0]])

    drone_body_y.set_xdata([states_history_x[num]+drone_pos_inertial_y_neg[0][0],states_history_x[num]+drone_pos_inertial_y[0][0]])
    drone_body_y.set_ydata([states_history_y[num]+drone_pos_inertial_y_neg[1][0],states_history_y[num]+drone_pos_inertial_y[1][0]])

    real_trajectory.set_xdata(states_history_x[0:num])
    real_trajectory.set_ydata(states_history_y[0:num])
    real_trajectory.set_3d_properties(states_history_z[0:num])

    drone_body_x.set_3d_properties([states_history_z[num]+drone_pos_inertial_x_neg[2][0],states_history_z[num]+drone_pos_inertial_x[2][0]])
    drone_body_y.set_3d_properties([states_history_z[num]+drone_pos_inertial_y_neg[2][0],states_history_z[num]+drone_pos_inertial_y[2][0]])

    drone_position_x.set_data(time_animation[0:num],states_history_x[0:num])
    drone_position_y.set_data(time_animation[0:num],states_history_y[0:num])
    drone_position_z.set_data(time_animation[0:num],states_history_z[0:num])
    drone_orientation_phi.set_data(time_animation[0:num],states_history_phi[0:num])
    drone_orientation_theta.set_data(time_animation[0:num],states_history_theta[0:num])
    drone_orientation_psi.set_data(time_animation[0:num],states_history_psi[0:num])

    return drone_body_x, drone_body_y, real_trajectory,\
    drone_position_x, drone_position_y, drone_position_z,\
    drone_orientation_phi, drone_orientation_theta, drone_orientation_psi

# Set up your figure properties
fig_x=16
fig_y=9
fig=plt.figure(figsize=(fig_x,fig_y),dpi=120,facecolor=(0.8,0.8,0.8))
n=4
m=3
gs=gridspec.GridSpec(n,m)

# Drone motion

# Create an object for the drone
ax0=fig.add_subplot(gs[0:3,0:2],projection='3d',facecolor=(0.9,0.9,0.9))
ax0.set_title('UAV Trajectory Tracking',fontsize=14)

# Plot the reference trajectory
ref_trajectory=ax0.plot(x_ref,y_ref,z_ref,'b',linewidth=1,label='Reference trajectory')
real_trajectory,=ax0.plot([],[],[],'r',linewidth=1,label='UAV trajectory')
drone_body_x,=ax0.plot([],[],[],'r',linewidth=5)
drone_body_y,=ax0.plot([],[],[],'g',linewidth=5)

ax0.set_xlim(min_ref,max_ref)
ax0.set_ylim(min_ref,max_ref)
ax0.set_zlim(0,max(z_ref))

ax0.set_xlabel('X [m]')
ax0.set_ylabel('Y [m]')
ax0.set_zlabel('Z [m]')
ax0.legend(loc='upper left')

# Drone position: X
ax1=fig.add_subplot(gs[3,0],facecolor=(0.9,0.9,0.9))
ax1.plot(time_trajectory,x_ref,'b',linewidth=1,label='X_ref [m]')
drone_position_x,=ax1.plot([],[],'r',linewidth=1,label='X [m]')
ax1.set_xlim(0,time_animation[-1])
ax1.set_ylim(np.min(states_history_x)-0.01,np.max(states_history_x)+0.01)
ax1.legend(loc='lower right',fontsize='small')
plt.grid(True)
plt.xlabel('t-time [s]',fontsize=15)

# Drone position: Y
ax2=fig.add_subplot(gs[3,1],facecolor=(0.9,0.9,0.9))
ax2.plot(time_trajectory,y_ref,'b',linewidth=1,label='Y_ref [m]')
drone_position_y,=ax2.plot([],[],'r',linewidth=1,label='Y [m]')
ax2.set_xlim(0,time_animation[-1])
ax2.set_ylim(np.min(states_history_y)-0.01,np.max(states_history_y)+0.01)
ax2.legend(loc='lower right',fontsize='small')
plt.grid(True)
plt.xlabel('t-time [s]',fontsize=15)

# Drone position: Z
ax3=fig.add_subplot(gs[3,2],facecolor=(0.9,0.9,0.9))
ax3.plot(time_trajectory,z_ref,'b',linewidth=1,label='Z_ref [m]')
drone_position_z,=ax3.plot([],[],'r',linewidth=1,label='Z [m]')
plt.xlim(0,time_animation[-1])
plt.ylim(np.min(states_history_z)-0.01,np.max(states_history_z)+0.01)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')
plt.xlabel('t-time [s]',fontsize=15)

# Create the function for Phi
ax4=fig.add_subplot(gs[0,2],facecolor=(0.9,0.9,0.9))
ax4.plot(time_mpc,angles_ref_history[:,0],'b',linewidth=1,label='Phi_ref [rad]')
drone_orientation_phi,=ax4.plot([],[],'r',linewidth=1,label='Phi [rad]')
plt.xlim(0,time_animation[-1])
plt.ylim(np.min(states_history_phi)-0.01,np.max(states_history_phi)+0.01)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

# Create the function for Theta
ax5=fig.add_subplot(gs[1,2],facecolor=(0.9,0.9,0.9))
ax5.plot(time_mpc,angles_ref_history[:,1],'b',linewidth=1,label='Theta_ref [rad]')
drone_orientation_theta,=ax5.plot([],[],'r',linewidth=1,label='Theta [rad]')
plt.xlim(0,time_animation[-1])
plt.ylim(np.min(states_history_theta)-0.01,np.max(states_history_theta)+0.01)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

# Create the function for Psi
ax6=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
ax6.plot(time_mpc,angles_ref_history[:,2],'b',linewidth=1,label='Psi_ref [rad]')
drone_orientation_psi,=ax6.plot([],[],'r',linewidth=1,label='Psi [rad]')
plt.xlim(0,time_animation[-1])
plt.ylim(np.min(states_history_psi)-0.01,np.max(states_history_psi)+0.01)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

drone_ani=animation.FuncAnimation(fig, update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)
plt.show()
