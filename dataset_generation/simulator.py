"""
Code to simulate a glider in a wind field.

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


class Simulator:
    """Class that represents a simulator that simulates a glider in a wind field."""

    _RESULT_NAMES = ("p_n_gl",   # Position of glider north
                     "p_e_gl",   # Position of glider east
                     "p_d_gl",   # Position of glider down
                     "v_n_gl",   # Velocity of glider north
                     "v_e_gl",   # Velocity of glider east
                     "v_d_gl",   # Velocity of glider down
                     "w_th",  # Velocity of updraft
                     "L_th",  # Roll moment induced by updraft
                     "t")     # Time

    def __init__(self, config):
        """Creates a new simulator that simulates a glider in a wind field."""
        # Load config relevant to the simulation
        self.config = config["simulation"]

        # Initialize random number generator
        self.rng = np.random.default_rng()

    def simulate(self, glider, wind):
        """Runs the simulation and returns the results as dictionary."""
        # Create dictionary to store result
        result = {k: [] for k in self._RESULT_NAMES}

        # Run simulation
        t_i = 0
        is_tracking_path = True
        while is_tracking_path:
            # Perform one simulation step
            is_tracking_path = glider.step()

            # Stop simulation if end of path is reached
            if not is_tracking_path:
                break
            else:
                # Get new state of glider
                position_i = glider.x[:3]
                velocity_i = glider.x[3:]
                phi_i = glider.control[0]  # Use commanded bank angle as approximation for the roll angle
                psi_i = glider.chi  # Use chi as approximation for psi since beta is assumed to be zero
                v_A_norm_i = glider.v_A_norm

                # Get wind data
                updraft_velocity_i = -wind.get_vertical_wind(position_i[:2])
                induced_roll_moment_i = wind.get_induced_roll_moment(position_i[:2], v_A_norm_i, phi_i, psi_i)

                # Add noise?
                if self.config["add_noise"]:
                    updraft_velocity_i += self.rng.normal(0, self.config["noise_sd_w"])
                    induced_roll_moment_i += self.rng.normal(0, self.config["noise_sd_l"])

                # Update time
                t_i += self.config["dt_simulation"]

                # Update data lists
                result["p_n_gl"].append(position_i[0])   # Position north
                result["p_e_gl"].append(position_i[1])   # Position east
                result["p_d_gl"].append(position_i[2])   # Position down
                result["v_n_gl"].append(velocity_i[0])   # Velocity north
                result["v_e_gl"].append(velocity_i[1])   # Velocity east
                result["v_d_gl"].append(velocity_i[2])   # Velocity down
                result["w_th"].append(updraft_velocity_i)     # Velocity updraft
                result["L_th"].append(induced_roll_moment_i)  # Induced roll moment
                result["t"].append(t_i)  # Time

        return result
