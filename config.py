import argparse
from os.path import expanduser, join
import os


def set_args():
    parser = argparse.ArgumentParser()
    source_path = join(expanduser("~"), "xyx", "data")
    data_path = os.path.join(expanduser("~"), "cvdata", "desert")
    # Basic Information
    # parser.add_argument('--name', type=str, default="test_sf")
    # parser.add_argument('--name', type=str, default="1:2:1_Regression_sigmoid_shuffle_score_debug_dropout_lmd1.5")
    parser.add_argument('--name', type=str, default="attention")
    parser.add_argument('--train_path', type=str, default=data_path)
    parser.add_argument('--test_path', type=str, default=" ")
    parser.add_argument('--submit_dir', type=str, default="submission")

    # Training Settings
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sample_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--n_eval', type=int, default=512)
    parser.add_argument('--lambda', type=float, default=1.5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--Regression', action='store_true', default=True)
    parser.add_argument('--Attention', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=False)  # If use small data 'desert'

    # Optimizer Settings
    parser.add_argument('--lr', type=float, default=1e-5)  # 8e-4
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--grad_clipping', type=float, default=5)

    # Inference Settings
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--update_rd', type=int, default=10)
    parser.add_argument('--test_img', type=str, default='resource/test_1.png')
    config = parser.parse_args().__dict__
    return config




