from model import *
import torch
from utils.Batcher import BatchGenerator
from utils.util import *
from tqdm import tqdm
from config import set_args
from utils.logger import create_logger
from tensorboardX import SummaryWriter


def predict():
    raise NotImplementedError


if __name__ == "__main__":
    logger = create_logger(__name__)
    config = set_args()
    global_step = 0
    writer = SummaryWriter(log_dir="figures")

    train_data = load_data(config['train_path'])
    test_data = load_data(config['test_path'])

    Batcher = BatchGenerator(config, train_data)
    model = BackgroundMatcher()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 betas=(config['b1'], config['b2']),
                                 eps=config['e'],
                                 weight_decay=config['decay'])
    criterion = nn.MSELoss(reduction=False)

    for epc in range(config['epochs']):
        Batcher.reset()
        for i, batch in tqdm(enumerate(Batcher), total=len(Batcher)):
            global_step += 1
            # Just for test
            xb = torch.rand(5, 3, 255, 255)
            xp = torch.rand(5, 3, 255, 255)
            xs = torch.rand(5, 3, 255, 255)

            y_pred = model(xb, xp, xs)

            loss = criterion(y_pred, batch['label'])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if i % 500 == 0:
                predict()
