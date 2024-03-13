"""
Code that models thermal updrafts and their effect on a glider.

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


class Updraft:
    """
    Class that represents thermal updrafts.

    Each updraft is defined by its strength, width, and position.

    Model from:
        Oettershagen, Philipp, et al.,
        "Robotic technologies for solar‐powered UAVs: fully autonomous updraft‐aware aerial sensing for multiday
        search‐and‐rescue missions",
        Journal of Field Robotics, 2018
        https://doi.org/10.1002/rob.21765
    """

    def __init__(self, config, pos_existing_updrafts=None):
        """
        Creates a new updraft object.

        The updraft parameters are randomly sampled from uniform distributions of the ranges specified in config.
        Provide positions of existing updrafts to ensure that the minimum distance between two updrafts is exceeded.
        """
        # Store config
        self.config = config

        # Initialize random number generator
        self.rng = np.random.default_rng()

        # Sample strength and width
        self.w = self.rng.uniform(*self.config["updrafts"]["w_range"])
        self.r = self.rng.uniform(*self.config["updrafts"]["r_range"])

        # Sample position. Optionally, ensure that minimum distance to existing updrafts is exceeded.
        self.pos_range = np.array(self.config["updrafts"]["pos_range"])
        if pos_existing_updrafts:
            self.pos = self.get_free_updraft_pos(pos_existing_updrafts)
        else:
            self.pos = self.rng.uniform(self.pos_range[:, 0], self.pos_range[:, 1])

    def get_free_updraft_pos(self, existing_positions):
        """Samples a new updraft position until it exceeds the minimum distance to existing updrafts."""
        if not isinstance(existing_positions, np.ndarray):
            existing_positions = np.asarray(existing_positions)

        while True:
            pos_candidate = self.rng.uniform(self.pos_range[:, 0], self.pos_range[:, 1])
            distances = np.linalg.norm(existing_positions - pos_candidate, axis=1)

            if np.greater(distances, self.config["updrafts"]["dist_min"]).all():
                break

        return pos_candidate

    def get_vertical_wind(self, pos_glider_2d):
        """Returns the vertical wind (aka the updraft strength) experienced by a glider at the specified 2D position."""
        delta = self.pos - pos_glider_2d  # Vector from glider position to updraft position
        return -self.w * np.exp(-(delta[0]**2 + delta[1]**2) / self.r**2)

    def get_induced_roll_moment(self, pos_glider_2d, v_a_glider, phi_glider, psi_glider):
        """Returns the roll moment induced by the updraft."""
        b = np.sqrt(self.config["glider"]["s"] * self.config["glider"]["st"])  # Wingspan of glider
        c = self.config["glider"]["s"] / b  # Mean chord of glider's wing
        delta = self.pos - pos_glider_2d  # Vector from glider position to updraft position

        return -(1 / 12) * self.config["glider"]["derivative_cl_alpha"] * self.config["physics"]["rho"] \
            * v_a_glider * c * b**3 * (self.w / self.r**2) \
            * np.exp(-(delta[0]**2 + delta[1]**2) / self.r**2) * np.cos(phi_glider) \
            * (np.cos(psi_glider) * delta[1] - np.sin(psi_glider) * delta[0])

    def get_info(self):
        """Returns the properties of the updraft (position, strength, width)."""
        return np.array((self.pos[0], self.pos[1], self.w, self.r))


class Wind:
    """Class that represents a wind field."""

    def __init__(self, config):
        """Creates a new wind object."""
        # Initialize random number generator
        self.rng = np.random.default_rng()

        # Add random number of updrafts
        self.n_updrafts = self.rng.choice(np.arange(config["updrafts"]["n"][0], config["updrafts"]["n"][1]+1),
                                          p=config["updrafts"]["probs_n"])
        self.updrafts = []
        updraft_positions = []
        for _ in range(self.n_updrafts):
            updraft = Updraft(config, updraft_positions)
            self.updrafts.append(updraft)
            updraft_positions.append(updraft.pos)

        # Store parameters for the horizontal wind
        self.v_horizontal_wind = np.array(config["horizontal_wind"]["v"])

    def get_vertical_wind(self, pos_glider_2d):
        """Returns the vertical wind that the glider experiences."""
        v_vertical_wind = 0
        for updraft in self.updrafts:
            v_vertical_wind += updraft.get_vertical_wind(pos_glider_2d)
        return v_vertical_wind

    def get_wind(self, pos_glider_2d):
        """Returns the wind experienced by a glider at the specified 2D position."""
        return np.hstack((self.v_horizontal_wind, self.get_vertical_wind(pos_glider_2d)))

    def get_induced_roll_moment(self, pos_glider_2d, v_a_glider, phi_glider, psi_glider):
        """Returns the roll moment induced by updrafts that the glider experiences."""
        induced_roll_moment = 0
        for updraft in self.updrafts:
            induced_roll_moment += updraft.get_induced_roll_moment(pos_glider_2d, v_a_glider, phi_glider, psi_glider)
        return induced_roll_moment
