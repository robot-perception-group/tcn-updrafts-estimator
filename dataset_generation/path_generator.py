"""
Code to generate flight paths.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import json
import numpy as np


class PathGenerator:
    """Class to generate flight paths."""

    def __init__(self, config):
        """Initializes a new path generator."""
        self.config = config["path"]
        self.north_border_width = int((self.config["field_size"][0]-self.config["flight_area_size"][0])/2)
        self.east_border_width = int((self.config["field_size"][1]-self.config["flight_area_size"][1])/2)

    def get_waypoints(self):
        """Calculates the waypoints of a path showing a zigzag lawnmower pattern and returns them as a list of lists."""
        waypoints = [[self.north_border_width, self.east_border_width+self.config["delta_from_path"]]]
        north_last, east_last = waypoints[0]
        direction_long = "north"  # "north", "south", "skip_then_north", or "skip_then_south"
        direction_lat = "west"  # "east" or "west"

        while True:
            # Step in east-west-direction
            if direction_long in ("skip_then_north", "skip_then_south"):
                if direction_lat == "east":
                    east_new = east_last + self.config["delta_east"] - 2*self.config["delta_from_path"]
                else:  # direction_lat == "west"
                    east_new = east_last + self.config["delta_east"] + 2*self.config["delta_from_path"]
            elif direction_lat == "east":
                east_new = east_last + 2 * self.config["delta_from_path"]
                direction_lat = "west"
            else:  # direction_lat == "west"
                east_new = east_last - 2 * self.config["delta_from_path"]
                direction_lat = "east"

            # Step in north-south-direction
            if direction_long == "north":
                north_new = north_last + self.config["delta_north"]
                if north_new + self.config["delta_north"] > self.config["field_size"][0] - self.north_border_width:
                    direction_long = "skip_then_south"
                    if direction_lat == "east":
                        direction_lat = "west"
                    else:
                        direction_lat = "east"
            elif direction_long == "south":
                north_new = north_last - self.config["delta_north"]
                if north_new - self.config["delta_north"] < self.north_border_width:
                    direction_long = "skip_then_north"
                    if direction_lat == "east":
                        direction_lat = "west"
                    else:
                        direction_lat = "east"
            elif direction_long == "skip_then_south":
                north_new = north_last
                direction_long = "south"
            else:  # direction_long == "skip_then_north"
                north_new = north_last
                direction_long = "north"

            # Check if done
            if east_new > self.config["field_size"][1] - self.east_border_width + self.config["delta_from_path"]:
                break

            # Add waypoint
            waypoints.append([north_new, east_new])

            # Remember last coordinates
            north_last, east_last = north_new, east_new

        return np.array(waypoints)

    @staticmethod
    def rotate_90_deg(waypoints):
        """Takes a path, rotates it by 90 degrees, and outputs it as a new path."""
        waypoints_rotated = waypoints.copy()
        waypoints_rotated = np.fliplr(waypoints_rotated)
        return waypoints_rotated

    @staticmethod
    def flip_horizontally(waypoints, field_width=1000):
        """Takes a path, flips it horizontally, and outputs it as a new path."""
        waypoints_flipped = waypoints.copy()
        waypoints_flipped[:, 1] = -waypoints_flipped[:, 1] + field_width
        return waypoints_flipped

    @staticmethod
    def flip_vertically(waypoints, field_width=1000):
        """Takes a path, flips it vertically, and outputs it as a new path."""
        waypoints_flipped = waypoints.copy()
        waypoints_flipped[:, 0] = -waypoints_flipped[:, 0] + field_width
        return waypoints_flipped

    @staticmethod
    def save_waypoints_as_json_file(waypoints, filepath):
        """Saves a list of lists with waypoints to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(waypoints.tolist(), f)
