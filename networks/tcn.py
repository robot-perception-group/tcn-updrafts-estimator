"""
Code that defines the Temporal Convolutional Network (TCN).

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import math
import torch
import torch.nn as nn


class TCN(nn.Module):
    """A Temporal Convolutional Network (TCN) to process a time series of 7000 measurements."""

    def __init__(self, config):
        """Initializes a Temporal Convolutional Network (TCN) to process a time series of 7000 measurements."""
        super().__init__()

        # Load config relevant to the network initialization
        self.config = config["network"]
        self.sequence_length = config["simulation"]["output_sequence_length"]

        # Parameters of convolutional layers
        self.conv_layers_parameters = (
            {"in_channels": self.config["obs_dim"], "out_channels": self.config["emb_dim"], "kernel_size":  1, "stride": 1, "padding": 0, "dilation": 1},
            {"in_channels": self.config["emb_dim"], "out_channels": self.config["emb_dim"], "kernel_size": 10, "stride": 1, "padding": 0, "dilation": 1},
            {"in_channels": self.config["emb_dim"], "out_channels": self.config["emb_dim"], "kernel_size": 10, "stride": 2, "padding": 0, "dilation": 1},
            {"in_channels": self.config["emb_dim"], "out_channels": self.config["emb_dim"], "kernel_size": 10, "stride": 4, "padding": 0, "dilation": 1},
            {"in_channels": self.config["emb_dim"], "out_channels": self.config["emb_dim"], "kernel_size": 10, "stride": 8, "padding": 0, "dilation": 1},
            {"in_channels": self.config["emb_dim"], "out_channels": self.config["emb_dim"], "kernel_size": 10, "stride": 1, "padding": 0, "dilation": 1}
        )

        # Set activation function
        if self.config["activation"] == "relu":
            self.activation_fn = nn.ReLU()
            self.nonlinearity_name_for_init = "relu"
        elif self.config["activation"] == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
            self.nonlinearity_name_for_init = "leaky_relu"
        else:
            raise RuntimeError(f"activation function must be 'relu' or 'leaky_relu', not {config['activation']}")

        # Define conv layers
        self.conv_layers = nn.ModuleList()
        seq_length_i = self.sequence_length
        for conv_layer_parameters in self.conv_layers_parameters:
            seq_length_i = self.calc_output_length(seq_length_i, *list(conv_layer_parameters.values())[2:])
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(**conv_layer_parameters),
                nn.LayerNorm([self.config["emb_dim"], seq_length_i],
                             elementwise_affine=self.config["elementwise_affine_transformation"]),
                self.activation_fn
            ))

        # Define output layers
        self.output_layer = nn.Sequential(nn.Flatten(start_dim=1),
                                          nn.Linear(seq_length_i * self.config["emb_dim"],
                                                    8 * self.config["emb_dim"]),
                                          nn.LayerNorm(8 * self.config["emb_dim"],
                                                       elementwise_affine=self.config["elementwise_affine_transformation"]),
                                          self.activation_fn,

                                          nn.Linear(8 * self.config["emb_dim"],
                                                    4 * self.config["emb_dim"]),
                                          nn.LayerNorm(4 * self.config["emb_dim"],
                                                       elementwise_affine=self.config["elementwise_affine_transformation"]),
                                          self.activation_fn,

                                          nn.Linear(4 * self.config["emb_dim"], self.config["out_dim"]))

        # Initialize weights
        if self.config["init_weights_normal"]:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes the weights and biases of the TCN."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_normal_(module.weight,
                                          a=0.01,
                                          mode="fan_in",
                                          nonlinearity=self.nonlinearity_name_for_init)
            module.bias.data.zero_()

    @staticmethod
    def calc_output_length(input_length, kernel_size, stride=1, padding=0, dilation=1):
        """Calculates the output length of a 1D convolutional layer."""
        return math.floor((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def forward(self, x):
        """Performs the forward pass through the TCN."""
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.output_layer(x)

        return x
