"""
Code that provides tools for processing the training and testing data.

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
import pandas as pd
import os
from torch.utils.data import Dataset


class Normalizer:
    """Class to normalize features and labels/predictions to prepare them for the training/inference."""

    def __init__(self, config):
        """Initializes a normalizer object based on the provided config."""
        # Load config
        n_updrafts_max = config["updrafts"]["n"][1]
        config_norm = config["normalization"]

        # Set the desired range of the data after the normalization
        self.l = config_norm["output_range"][0]
        self.u = config_norm["output_range"][1]

        # Set min and max arrays for x
        self.min_x = np.array([minimum for minimum, maximum in config_norm["training_data_range"].values()])
        self.max_x = np.array([maximum for maximum, maximum in config_norm["training_data_range"].values()])

        # Set min and max arrays for y
        self.min_y = np.array([minimum for minimum, maximum in config_norm["labels_and_predictions_range"].values()])
        self.max_y = np.array([maximum for maximum, maximum in config_norm["labels_and_predictions_range"].values()])
        self.min_y = np.tile(self.min_y, n_updrafts_max)  # Take into account that values for n_updrafts_max can be in y
        self.max_y = np.tile(self.max_y, n_updrafts_max)

    def normalize_x(self, x):
        """Normalizes the input features."""
        return (x - self.min_x) / (self.max_x - self.min_x) * (self.u - self.l) + self.l

    def unnormalize_x(self, x):
        """Unnormalizes the input features to restore their true range."""
        includes_roll_moment = False if x.shape[1] == 3 else True
        if includes_roll_moment:
            columns_to_unnorm = np.array((0, 1, 2, 3))
        else:
            columns_to_unnorm = np.array((0, 1, 2))

        return (x[:, columns_to_unnorm] - self.l) / (self.u - self.l) * (self.max_x[columns_to_unnorm] - self.min_x[columns_to_unnorm]) + self.min_x[columns_to_unnorm]

    def normalize_y(self, y):
        """Normalizes the input labels/predictions."""
        y = y[0]
        indices_to_norm = (y != -1).nonzero()  # Do not scale minus ones that mark that there are no more updrafts
        y[indices_to_norm] = ((y[indices_to_norm] - self.min_y[indices_to_norm]) /
                               (self.max_y[indices_to_norm] - self.min_y[indices_to_norm]) *
                               (self.u - self.l) + self.l)
        return y

    def unnormalize_y(self, y):
        """Unnormalizes the input labels/predictions to restore their true range."""
        y = y[0]
        indices_to_unnorm = (y > -0.5).nonzero()  # Do not scale minus ones that mark that there are no more updrafts
        y[indices_to_unnorm] = (y[indices_to_unnorm] - self.l) / (self.u - self.l) * (self.max_y[indices_to_unnorm] - self.min_y[indices_to_unnorm]) + self.min_y[indices_to_unnorm]
        return y


class UpdraftsDataset(Dataset):
    """Class that represents a dataset for training an ML-based updraft estimator."""

    def __init__(self, x_dir, y_dir, x_transform=None, y_transform=None):
        """Initializes a dataset object for training an ML-based updraft estimator."""
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        """Returns the number of datapoints in the dataset."""
        return len([f for f in os.listdir(self.y_dir) if os.path.isfile(os.path.join(self.y_dir, f))])

    def __getitem__(self, idx):
        """Returns the features and the label of a datapoint from the dataset."""
        # Load and process features (x)
        x_path = os.path.join(self.x_dir, f"{idx}.csv")
        x = pd.read_csv(x_path)
        x = x.drop(columns=["t"])
        x = x.to_numpy()
        if self.x_transform:
            try:
                for t in self.x_transform:  # Multiple transform functions
                    x = t(x)
            except TypeError:  # Only one transform function
                x = self.x_transform(x)

        # Load and process label (y)
        y_path = os.path.join(self.y_dir, f"{idx}.csv")
        y = pd.read_csv(y_path)
        y = y.to_numpy()
        if self.y_transform:
            try:
                for t in self.y_transform:  # Multiple transform functions
                    y = t(y)
            except TypeError:  # Only one transform function
                y = self.y_transform(y)

        return x, y


def remove_roll_moment(x):
    """Removes the induced roll moment data from the input features."""
    return x[:, :-1]
