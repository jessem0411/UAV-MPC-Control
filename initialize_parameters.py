'''
File: initialize_parameters
Description: initialize all the parameters needed for the trajectory tracking of a UAV
'''
import numpy as np

class InitializeParameters:
    def __init__(self):
        # Constants (Astec Hummingbird)
        self.Ixx = 0.0034  # kg*m^2
        self.Iyy = 0.0034  # kg*m^2
        self.Izz = 0.006  # kg*m^2
        self.m = 0.698  # kg
        self.g = 9.81  # m/s^2
        self.Jtp = 1.302 * 10 ** (-6)  # N*m*s^2=kg*m^2
        self.sampling_time = 0.1  # s

        # Matrix weights for the cost function
        self.Q = np.matrix('10 0 0;0 10 0;0 0 10')  # weights for outputs
        self.S = np.matrix('20 0 0;0 20 0;0 0 20')  # weights for the final horizon period outputs
        self.R = np.matrix('10 0 0;0 10 0;0 0 10')  # weights for inputs

        # Thrust and torque coefficients
        self.Ct = 7.6184 * 10 ** (-8) * (60 / (2 * np.pi)) ** 2  # N*s^2
        self.Cq = 2.6839 * 10 ** (-9) * (60 / (2 * np.pi)) ** 2  # N*m*s^2

        # Reference length of drone
        self.l = 0.171  # m

        self.num_att_outputs = 3  # Number of attitude outputs: phi,theta,psi
        self.max_horizon = 4  # horizon period

        self.pos_x_y = 0  # Default: 0. Make positive x and y longer for visual purposes (1-Yes, 0-No). It does not affect the dynamics of the UAV.
        self.inner_loop_length = 4  # Number of inner control loop iterations

        # Set up the poles for the feedback linearization
        self.poles_x = np.array([-1, -2])
        self.poles_y = np.array([-1, -2])
        self.poles_z = np.array([-1, -2])

        # Constants for trajectories
        self.r = 2   # radius of spiral
        self.f = 0.025   # frequency
        self.height_i = 5   # initial height
        self.height_f = 25   # final height

        self.sub_loop = 5  # for animation purposes


        # Drag force switch:
        self.drag_switch = 1  # Must be either 0 or 1 (0 - drag force OFF, 1 - drag force ON)

        # Drag coefficients:
        self.C_D_u = 1.5
        self.C_D_v = 1.5
        self.C_D_w = 2.0

        # Drag force cross-section area [m^2]
        self.A_u = 2 * self.l * 0.01 + 0.05 ** 2
        self.A_v = 2 * self.l * 0.01 + 0.05 ** 2
        self.A_w = 2 * 2 * self.l * 0.01 + 0.05 ** 2

        # Air density
        self.rho = 1.225  # [kg/m^3]
        self.trajectory = 1  # Choose the trajectory: only from 1-9
        self.no_plots = 0  # 0-you will see the plots; 1-you will skip the plots (only animation)

        # Constraints
        self.omega_min = 110*np.pi/3
        self.omega_max = 860*np.pi/3