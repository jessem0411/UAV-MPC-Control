'''
File: trajectory_generation.py
Description: generate reference values of [x,y,z,xdot,ydot,zdot,xddot,yddot,zddot,psi] for the UAV to track
'''

import numpy as np

def trajectory_generation(drone_parameters, time_trajectory):
        sampling_time = drone_parameters.sampling_time
        inner_loop_length = drone_parameters.inner_loop_length
        r = drone_parameters.r
        f = drone_parameters.f
        height_i = drone_parameters.height_i
        height_f = drone_parameters.height_f
        trajectory = drone_parameters.trajectory
        delta_height = height_f-height_i

        # Define the x, y, z dimensions for the drone trajectories
        alpha = 2 * np.pi * f * time_trajectory     # angle for circular trajectories

        if trajectory==1 or trajectory==2 or trajectory==3 or trajectory==4:
            # Trajectory 1
            x = r*np.cos(alpha)
            y = r*np.sin(alpha)
            z = height_i+delta_height/(time_trajectory[-1])*time_trajectory

            x_dot = -r*np.sin(alpha)*2*np.pi*f
            y_dot = r*np.cos(alpha)*2*np.pi*f
            z_dot = delta_height/(time_trajectory[-1])*np.ones(len(time_trajectory))

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            z_dot_dot=0*np.ones(len(time_trajectory))

            if trajectory==2:
                # Trajectory 2
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(time_trajectory[101:len(time_trajectory)]-time_trajectory[100])/20+x[100]
                y[101:len(y)]=2*(time_trajectory[101:len(time_trajectory)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+delta_height/time_trajectory[-1]*(time_trajectory[101:len(time_trajectory)]-time_trajectory[100])

                x_dot[101:len(x_dot)]=1/10*np.ones(len(time_trajectory[101:len(time_trajectory)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(time_trajectory[101:len(time_trajectory)]))
                z_dot[101:len(z_dot*(time_trajectory/20))]=delta_height/(t[-1])*np.ones(len(time_trajectory[101:len(time_trajectory)]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(time_trajectory[101:len(time_trajectory)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(time_trajectory[101:len(time_trajectory)]))
                z_dot_dot[101:len(z_dot_dot)]=0*np.ones(len(time_trajectory[101:len(time_trajectory)]))


        # Vector of x and y changes per sample time
        dx=x[1:len(x)]-x[0:len(x)-1]
        dy=y[1:len(y)]-y[0:len(y)-1]
        dz=z[1:len(z)]-z[0:len(z)-1]

        dx=np.append(np.array(dx[0]),dx)
        dy=np.append(np.array(dy[0]),dy)
        dz=np.append(np.array(dz[0]),dz)


        # Define the reference yaw angles
        psi=np.zeros(len(x))
        psi_ref = psi
        psi[0]=np.arctan2(y[0],x[0])+np.pi/2
        psi[1:len(psi)]=np.arctan2(dy[1:len(dy)],dx[1:len(dx)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psi_ref[0]=psi[0]
        for i in range(1,len(psi_ref)):
            if dpsi[i-1]<-np.pi:
                psi_ref[i]=psi_ref[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psi_ref[i]=psi_ref[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psi_ref[i]=psi_ref[i-1]+dpsi[i-1]

        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psi_ref