"""
Code to generate the items of a dataset for ML-based thermal updraft detection.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import random
import numpy as np
import pandas as pd
from dataset_generation.simulator import Simulator


class DataGenerator:
    """Class to generate the items of a dataset for ML-based thermal updraft detection."""

    def __init__(self, config, wind_physics_class, glider_physics_class, waypoints, path_transformations=None):
        """Initiates a DataGenerator that generates the items of a dataset for ML-based thermal updraft detection."""
        # Store config, glider/wind physics, and waypoints
        self.config = config
        self.waypoints = waypoints
        self.path_transformations = path_transformations
        self.wind_model = wind_physics_class
        self.glider_model = glider_physics_class
        self.north_min, self.north_max = 0, config["path"]["field_size"][0]
        self.east_min, self.east_max = 0, config["path"]["field_size"][1]
        self.updraft_max = config["updrafts"]["w_range"][1]

        # Initialize simulator
        self.simulator = Simulator(config)

        # Calc sampling info
        self.steps_to_skip = int(config["simulation"]["dt_output_sequence"] / config["simulation"]["dt_simulation"])

        # Get max number of updrafts
        self.n_updrafts_max = config["updrafts"]["n"][1]

        # Prepare column names for labels
        self.column_names_labels = []
        for i in range(self.n_updrafts_max):
            self.column_names_labels += [f"p_n_th_{i}", f"p_e_th_{i}", f"w_th_{i}", f"r_th_{i}"]

    def generate_data(self, output_file_paths):
        """Generates one pair of features and corresponding labels and saves it as csv files."""
        # Run new simulations until the required sequence length is obtained
        while True:
            # Prepare waypoints
            waypoints = self.waypoints
            if self.path_transformations is not None:
                for path_transformation in self.path_transformations:
                    if random.choice([True, False]):
                        waypoints = path_transformation(waypoints)

            # Initialize new wind field and glider instance
            wind = self.wind_model(self.config)
            glider = self.glider_model(self.config, waypoints, wind.get_wind)

            # Simulate
            simulation_result = self.simulator.simulate(glider, wind)

            if (int(len(simulation_result["t"]) / self.steps_to_skip) >=
                    self.config["simulation"]["output_sequence_length"]):
                break

        # Get label (updraft ground truth)
        y = -np.ones((1, 4 * self.n_updrafts_max))  # output padding with -1
        updrafts_sorted = sorted(wind.updrafts, key=lambda u: u.w, reverse=True)  # Sort -> stronger updrafts first
        for i, updraft in enumerate(updrafts_sorted):  # Fill label variable
            y[0, 4*i:4*i + 4] = updraft.get_info()
        y = pd.DataFrame(y, columns=self.column_names_labels)

        # Get features (sequence of simulated sensor measurements)
        x = pd.DataFrame(simulation_result)
        x = x[["t", "p_n_gl", "p_e_gl", "w_th", "L_th"]]  # Pick relevant data
        x["t"] -= x["t"][0]  # Let t start at 0 s
        x = x[::self.steps_to_skip]  # Resample to match desired sample period of output sequences
        x = x.reset_index(drop=True)  # Reset index after resampling
        x = x[:self.config["simulation"]["output_sequence_length"]]  # Truncate too long sequences

        # Save to csv files
        y.to_csv(output_file_paths["y"], index=False, float_format='%.4E')
        x.to_csv(output_file_paths["x"], index=False, float_format='%.4E')
