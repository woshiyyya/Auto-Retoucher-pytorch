import argparse
from os.path import expanduser, join
import os


def set_args():
    parser = argparse.ArgumentParser()
    source_path = join(expanduser("~"), "xyx", "data")
    data_path = os.path.join(expanduser("~"), "cvdata", "desert")
    # Basic Information
    parser.add_argument('--name', type=str, default="default")
    parser.add_argument('--train_path', type=str, default=data_path)
    parser.add_argument('--test_path', type=str, default=" ")
    parser.add_argument('--submit_dir', type=str, default="submission")

    # Training Settings
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sample_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--n_eval', type=int, default=512)
    parser.add_argument('--lambda', type=float, default=1)
    parser.add_argument('--cuda', action='store_true', default=True)

    # Optimizer Settings
    parser.add_argument('--lr', type=float, default=4e-4)  # 8e-4
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--grad_clipping', type=float, default=5)
    config = parser.parse_args().__dict__
    return config
