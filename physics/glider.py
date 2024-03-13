"""
Code that models a glider flying in a wind field.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import numpy as np


class Glider:
    """Class that represents a glider, which is modeled with 3 DoF, flying in a wind field."""

    def __init__(self, config, waypoints, wind_fun):
        """
        Creates a new glider object.

        Provide the config, the waypoints of the desired flight path, and a function that returns the wind for a given
        glider position.
        """
        # Store config and wind function
        self.config = config
        self.wind_fun = wind_fun

        # Initialize random number generator
        self.rng = np.random.default_rng()

        # Process waypoints info
        self.waypoints = waypoints
        self.waypoint_id_max = len(waypoints) - 1
        self.waypoint_id_current = 0

        # Process control info
        self.alpha_cmd = np.deg2rad(self.config["control"]["alpha_cmd"])
        self.phi_max = np.deg2rad(self.config["control"]["phi_max"])
        self.control = np.array([0, self.alpha_cmd])

        # Calc initial velocity
        self.v_A_norm = self.rng.uniform(*self.config["control"]["V_A_range"])
        vec_to_second_wp = self.waypoints[1]-self.waypoints[0]
        vec_to_second_wp = vec_to_second_wp/np.linalg.norm(vec_to_second_wp)  # Normalize to get unit vector
        velocity_init = vec_to_second_wp * self.v_A_norm

        # Set initial state (Start at first waypoint, facing the second waypoint)
        self.x = np.array([self.waypoints[0][0],  # north position
                           self.waypoints[0][1],  # east position
                           -self.config["control"]["initial_altitude"],  # down position
                           velocity_init[0],      # north velocity
                           velocity_init[1],      # east velocity
                           0])                    # down velocity
        self.chi = np.arctan2(self.x[3], self.x[4])

        # Instantiate state derivative
        self.x_dot = np.zeros_like(self.x)

    def get_closest_waypoint(self):
        """Returns the waypoint that is currently closest to the glider."""
        distances = np.linalg.norm(self.waypoints - self.x[0:2], axis=1)
        closest_waypoint_id = np.argmin(distances)
        return self.waypoints[closest_waypoint_id]

    def get_current_waypoint(self):
        """Returns the current waypoint, i.e. the waypoint the glider is tasked to reach next."""
        return self.waypoints[self.waypoint_id_current]

    def update_waypoint_id(self):
        """Updates the waypoint id, i.e. defines which waypoint should be reached next."""
        if np.linalg.norm(self.get_current_waypoint() - self.x[0:2]) < self.config["control"]["dist_switch_wp"]:
            self.waypoint_id_current += 1

    def update_control(self):
        """Calculates the control commands."""
        # Get position and velocity
        position = self.x[0:3]
        velocity = self.x[3:6]

        # Calc chi
        self.chi = np.arctan2(velocity[1], velocity[0])

        # Calc guidance
        g_los = self.get_current_waypoint() - position[0:2]
        chi_cmd = np.arctan2(g_los[1], g_los[0])

        # Calc current error
        chi_error = chi_cmd - self.chi

        # Enforce that the angle error lies between -pi and +pi
        if chi_error > np.pi:
            chi_error -= (2 * np.pi)
        elif chi_error < -np.pi:
            chi_error += (2 * np.pi)

        # Evaluate the control law
        phi_cmd = self.config["control"]["k_chi"] * chi_error

        # Enforce control limits
        phi_cmd = np.clip(phi_cmd, -self.phi_max, self.phi_max)

        # Update control
        self.control[0] = phi_cmd
        self.control[1] = self.alpha_cmd

    def update_dynamics(self):
        """
        Calculates the state derivative x_dot.

        Model from:
            Notter, S., Schimpf, F., Müller, G. & Fichter, W.,
            "Hierarchical Reinforcement Learning Approach for Autonomous Cross-Country Soaring",
            Journal of Guidance, Control, and Dynamics, 2023.
            https://doi.org/10.2514/1.G006746
            https://github.com/ifrunistuttgart/RL_CrossCountrySoaring
        """
        # Control variables assignment
        mu_a = self.control.item(0)
        alpha = self.control.item(1)

        # Get wind vector at current aircraft position
        g_v_W = self.wind_fun(self.x[0:2])

        # Track speed in local NED coordinates
        g_v_K = self.x[3:6]

        # Airspeed in local NED coordinates: airspeed = groundspeed - windspeed
        g_v_A = g_v_K - g_v_W

        # Air-path angles
        self.v_A_norm = np.maximum(np.linalg.norm(g_v_A), .1)
        gamma_a = -np.arcsin(np.clip((g_v_A[2] / self.v_A_norm), -1, 1))
        chi_a = np.arctan2(g_v_A[1], g_v_A[0])

        # Aerodynamic force in aerodynamic coordinates
        cl = 2 * np.pi * (self.config["glider"]["st"] / (self.config["glider"]["st"] + 2)) * alpha
        cd = self.config["glider"]["cd0"] + (1 / (np.pi * self.config["glider"]["st"] * self.config["glider"]["oe"])) * np.power(cl, 2)
        a_f_A = (self.config["physics"]["rho"] / 2) * self.config["glider"]["s"] * np.power(self.v_A_norm, 2) * np.array(
            [[-cd], [0], [-cl]])

        # Aerodynamic force in local NED coordinates
        g_T_a = self.get_rotation_matrix(-chi_a.item(), 3) \
                @ self.get_rotation_matrix(-gamma_a.item(), 2) \
                @ self.get_rotation_matrix(-mu_a, 1)
        g_f_A = g_T_a @ a_f_A

        # Track acceleration in local NED coordinates
        g_a_K = (g_f_A / self.config["glider"]["m"]) + np.array([[0], [0], [self.config["physics"]["g"]]])

        # State derivative
        self.x_dot = np.append(g_v_K, g_a_K)

    def step(self):
        """
        Performs one Euler integration step.

        If the end of the path is reached, False is returned, otherwise True.
        """
        # Select waypoint for tracking
        self.update_waypoint_id()
        if self.waypoint_id_current > self.waypoint_id_max:
            return False

        # Update control and dynamics
        self.update_control()
        self.update_dynamics()

        # Integrate (Euler method)
        self.x += self.config["simulation"]["dt_simulation"] * self.x_dot

        # Return True to indicate that the end of the path has not been reached yet.
        return True

    @staticmethod
    def get_rotation_matrix(angle, axis):
        """
        Returns the rotation matrix for a rotation with the specified angle around the selected axis.

        (x = 1, y = 2, z = 3)
        """
        if axis == 1:
            rotation_matrix = np.array([[1, 0, 0],
                                       [0, np.cos(angle), np.sin(angle)],
                                       [0, -np.sin(angle), np.cos(angle)]])
        elif axis == 2:
            rotation_matrix = np.array([[np.cos(angle), 0, -np.sin(angle)],
                                       [0, 1, 0],
                                       [np.sin(angle), 0, np.cos(angle)]])
        else:  # axis == 3:
            rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0],
                                       [-np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1]])
        return rotation_matrix
