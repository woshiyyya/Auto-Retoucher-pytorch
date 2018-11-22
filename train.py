from model import *
import torch
from utils.Batcher import BatchGenerator
from utils.util import *
from tqdm import tqdm
from config import set_args
from utils.logger import create_logger
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    logger = create_logger(__name__)
    config = set_args()
    global_step = 0
    writer = SummaryWriter(log_dir="figures")

    bg_data, fg_data, sp_data, sf_data = load_data()
    # test_data = load_data(config['test_path'])

    Batcher = BatchGenerator(config, bg_data, fg_data, sp_data, sf_data)
    print(Batcher.total)
    model = BackgroundMatcher()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 betas=(config['b1'], config['b2']),
                                 eps=config['e'],
                                 weight_decay=config['decay'])
    criterion = nn.CrossEntropyLoss(reduce=False)

    if config['cuda']:
        model.cuda()

    for epc in range(config['epochs']):
        Batcher.reset()
        for i, batch in tqdm(enumerate(Batcher), total=len(Batcher)):
            global_step += 1
            # Just for test
            y1_pred, y2_pred = model(batch)

            loss1 = torch.sum(criterion(y1_pred, batch['y1']))
            loss2 = torch.sum(criterion(y2_pred, batch['y2']))
            loss = loss1 + config['lambda'] * loss2
            loss.backward()
            if i % 500 == 0:
                add_figure(config['name'], writer, global_step, loss1, loss2, loss)
                print(loss.detach().cpu().numpy())
            optimizer.step()
            optimizer.zero_grad()
        torch.save(model.state_dict(), "checkpoints/ckpt_epoch{}.pth".format(epc))
    writer.close()
