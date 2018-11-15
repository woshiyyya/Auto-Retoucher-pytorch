from model import *
import torch
import argparse
from utils.Batcher import BatchGenerator
from utils.util import *
from tqdm import tqdm


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="default")
    parser.add_argument('--train_path', type=str, default=" ")
    parser.add_argument('--test_path', type=str, default=" ")
    parser.add_argument('--submit_dir', type=str, default="submission")
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--epochs', type=int, default=10)
    config = parser.parse_args().__dict__

    train_data = load_data(config['train_path'])
    test_data = load_data(config['test_path'])

    Batcher = BatchGenerator(config, train_data)
    model = BackgroundMatcher()

    for i in range(config['epochs']):
        for batch in tqdm(Batcher)
            xb = torch.rand(5, 3, 255, 255)
            xp = torch.rand(5, 3, 255, 255)
            xs = torch.rand(5, 3, 255, 255)
            model(xb, xp, xs)
