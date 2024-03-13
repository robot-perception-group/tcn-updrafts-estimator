"""
Tool to train a TCN-based updraft estimator.

Copyright (c) 2024 Institute of Flight Mechanics and Controls

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
https://www.gnu.org/licenses.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from utils.data import UpdraftsDataset, Normalizer, remove_roll_moment
from networks.tcn import TCN


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Tool to train a TCN-based updraft estimator.")
    parser.add_argument("config",
                        help="File path to a YAML file containing the config")
    parser.add_argument("--dataset_dir", default=None,
                        help="Directory where the dataset is stored, default: Use directory specified in config")
    parser.add_argument("--train_folder", default="train",
                        help="Name of the folder in dataset_dir that contains the training data, default: 'train'")
    parser.add_argument("--val_folder", default="val",
                        help="Name of the folder in dataset_dir that contains the validation data, default: 'val'")
    parser.add_argument("--x_folder", default="x",
                        help="Name of the folders in dataset_dir/... that contain the features, default: 'x'")
    parser.add_argument("--y_folder", default="y",
                        help="Name of the folders in dataset_dir/... that contain the labels, default: 'y'")
    parser.add_argument("--checkpoints_folder", default="checkpoints",
                        help="Name of the folder where the checkpoints will be saved, default: 'checkpoints'")
    parser.add_argument("--models_folder", default="models",
                        help="Name of the folder where the models will be saved, default: 'models'")
    parser.add_argument("--device", default=None,
                        help="Device used for training, default: use GPU if available")
    return parser.parse_args()


def train_loop(dataloader, model, loss_fn, optimizer, device, return_example=False, gradient_clip=False):
    """Performs one training epoch."""
    size = len(dataloader.dataset)
    model.train()
    loss_train = 0
    for x, y in dataloader:
        # Move data to device
        x, y = x.float().to(device), y.float().to(device)

        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_train += loss * len(x)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

    loss_train /= size

    if return_example:
        return loss_train, pred[0], y[0]
    else:
        return loss_train


def val_loop(dataloader, model, loss_fn, device, return_example=False):
    """Performs one validation run."""
    model.eval()
    size = len(dataloader.dataset)
    loss_val = 0

    with torch.no_grad():
        for x, y in dataloader:
            # Move data to device
            x, y = x.float().to(device), y.float().to(device)

            # Compute prediction and loss
            pred = model(x)
            loss = loss_fn(pred, y)
            loss_val += loss * len(x)

    loss_val /= size

    if return_example:
        return loss_val, pred[0], y[0]
    else:
        return loss_val


def main(args):
    # Set device for PyTorch
    if args.device:
        device = args.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set writer for TensorBoard
    writer = SummaryWriter(os.path.join(os.getcwd(), os.path.join("runs", f"{config['training']['experiment_name']}")))

    # Prepare directories
    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.models_folder, exist_ok=True)

    # Init data normalization
    normalizer = Normalizer(config)

    # Configure dataset
    if config["training"]["use_roll_moment_data"]:
        x_transform_functions = (normalizer.normalize_x, np.transpose)
    else:
        x_transform_functions = (normalizer.normalize_x, remove_roll_moment, np.transpose)
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else config["training"]["dataset_dir"]
    data_train = UpdraftsDataset(os.path.join(dataset_dir, args.train_folder, args.x_folder),
                                 os.path.join(dataset_dir, args.train_folder, args.y_folder),
                                 x_transform_functions, normalizer.normalize_y)
    data_val = UpdraftsDataset(os.path.join(dataset_dir, args.val_folder, args.x_folder),
                               os.path.join(dataset_dir, args.val_folder, args.y_folder),
                               x_transform_functions, normalizer.normalize_y)

    # Configure dataloader
    dataloader_train = DataLoader(data_train, batch_size=config["training"]["mini_batch_size"], shuffle=True)
    dataloader_val = DataLoader(data_val, batch_size=config["training"]["mini_batch_size"], shuffle=True)

    # Configure network
    model = TCN(config).to(device)

    # Define loss function
    loss_fn = nn.MSELoss()

    # Set optimizer settings
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["training"]["lr"],
                                 weight_decay=config["training"]["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["training"]["lr_decay_steps"], gamma=0.1)

    # Train and validate
    print("Training started. Progress:")
    for epoch in tqdm(range(config["training"]["epochs"])):
        loss_train, example_pred_train, example_label_train = train_loop(dataloader_train, model, loss_fn, optimizer,
                                                                         device, return_example=True,
                                                                         gradient_clip=config["training"]["gradient_clipping"])
        loss_val, example_pred_val, example_label_val = val_loop(dataloader_val, model, loss_fn,
                                                                 device, return_example=True)

        # Send info to TensorBoard
        writer.add_scalar(f"Loss/train", loss_train, epoch)
        writer.add_scalar(f"Loss/val", loss_val, epoch)
        writer.add_scalar(f"Lr", lr_scheduler.get_last_lr()[0], epoch)
        example_label_train_str = np.array2string(example_label_train.detach().cpu().numpy(), precision=2, floatmode="fixed")
        example_pred_train_str = np.array2string(example_pred_train.detach().cpu().numpy(), precision=2, floatmode="fixed")
        example_label_val_str = np.array2string(example_label_val.cpu().numpy(), precision=2, floatmode="fixed")
        example_pred_val_str = np.array2string(example_pred_val.cpu().numpy(), precision=2, floatmode="fixed")
        writer.add_text(f"Examples/train",
                        f"True: \n {example_label_train_str}  \n Pred: \n {example_pred_train_str}", epoch)
        writer.add_text(f"Examples/val",
                        f"True: \n {example_label_val_str}  \n Pred: \n {example_pred_val_str}", epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': loss_train},
                   f"{args.checkpoints_folder}/latest_checkpoint_{config['training']['experiment_name']}.pth")

        lr_scheduler.step()

    # Close TensorBoard writer
    writer.flush()
    writer.close()

    # Save model
    torch.save(model.state_dict(), f"{args.models_folder}/model_{config['training']['experiment_name']}.pth")

    print("Training completed.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
