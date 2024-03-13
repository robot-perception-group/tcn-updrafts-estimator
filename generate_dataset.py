"""
Tool to generate a dataset for training an ML-based updraft estimator.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

from multiprocessing import Pool
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml
from dataset_generation.data_generator import DataGenerator
from dataset_generation.path_generator import PathGenerator
from physics.wind import Wind
from physics.glider import Glider


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Tool to generate a dataset for training an ML-based updraft estimator.")
    parser.add_argument("config",
                        help="File path to a YAML file containing the config")
    parser.add_argument("waypoints",
                        help="File path to a JSON file containing a list of waypoints")
    parser.add_argument("--output_dir", default="",
                        help="Directory where the dataset is saved, default: in working directory")
    parser.add_argument("--dataset_name", default="dataset",
                        help="Name of the dataset, default: 'dataset'")
    parser.add_argument("--sub_folder", default="",
                        help="Sub folder, e.g., for train/val/test splits, default: no sub folder")
    parser.add_argument("--index_start", type=int, default=0,
                        help="Index of the first output dataset element, default: 0")
    parser.add_argument("--index_end", type=int, default=10000,
                        help="Index of the final output dataset element, (index_end+1 - index_start) defines the number "
                             "of generated dataset elements, default: 10000")
    parser.add_argument("--processes", type=int, default=os.cpu_count(),
                        help="Number of processes, default: use all cpu cores")
    parser.add_argument('--transform_path', action="store_true",
                        help="Use this flag to randomly flip and rotate the generated paths.")
    return parser.parse_args()


def main(args):
    # Check if indices are valid
    assert args.index_start < args.index_end

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load waypoints
    with open(args.waypoints) as f:
        waypoints_list = json.load(f)
    waypoints = np.array(waypoints_list)

    # Initialize data generator
    if args.transform_path:
        path_transformations = (PathGenerator.rotate_90_deg, PathGenerator.flip_vertically, PathGenerator.flip_horizontally)
    else:
        path_transformations = None
    data_generator = DataGenerator(config, Wind, Glider, waypoints, path_transformations=path_transformations)

    # Prepare directories
    dir_x = os.path.join(args.output_dir, args.dataset_name, args.sub_folder, "x")
    dir_y = os.path.join(args.output_dir, args.dataset_name, args.sub_folder, "y")
    os.makedirs(dir_x, exist_ok=True)
    os.makedirs(dir_y, exist_ok=True)

    # Create list of output file paths
    output_file_paths = [{"x": os.path.join(dir_x, f"{str(i)}.csv"),
                          "y": os.path.join(dir_y, f"{str(i)}.csv")}
                         for i in range(args.index_start, args.index_end+1)]

    # Generate dataset using parallel computing
    print("Dataset generation started. Progress:")
    with Pool(args.processes) as pool:
        for _ in tqdm(pool.imap(data_generator.generate_data, output_file_paths),
                      total=len(output_file_paths), smoothing=0.001):
            pass
    print("Dataset generation completed.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
