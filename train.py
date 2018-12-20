import torch
from torch import nn
from utils.batcher import BatchGenerator
from utils.util import *
from tqdm import tqdm
from config import set_args
from utils.logger import create_logger
from tensorboardX import SummaryWriter
from model.verifier import Verifier
from model.verifier_base import VerifierBase


def predict(batcher, model):
    test_data = batcher.test_batches
    model.eval()
    acc = 0.0
    rmse = []
    for i, batch in tqdm(enumerate(test_data)):
        batch = batcher.batch2cuda(batch)
        y1 = batch['y1'].detach().cpu().numpy()
        y2 = batch['y2'].detach().cpu().numpy()
        y1_pred, y2_pred = model(batch)
        y1_pred, y2_pred = y1_pred.detach().cpu().numpy()[:, 0], y2_pred.detach().cpu().numpy()[:, 0]
        y1_pred[y1_pred >= 0.5] = 1
        y1_pred[y1_pred < 0.5] = 0
        acc += np.sum(y1 == y1_pred) / y1_pred.shape[0]
        rmse.append((y2_pred - y2) ** 2)
    acc /= len(test_data)
    rmse = np.sqrt(np.mean(np.concatenate(rmse)))
    model.train()
    return float(acc), float(rmse)


def print_acc(batch, y1_pred, y2_pred, step):
    global total_acc1, total_acc2

    acc1, acc2 = accuracy(batch, y1_pred, y2_pred)
    total_acc1 += acc1
    total_acc2 += acc2
    print("batch_acc:", acc1, acc2)
    print("total_acc:", total_acc1/step, total_acc2/step)


if __name__ == "__main__":
    logger = create_logger(__name__)
    config = set_args()
    global_step = 0
    writer = SummaryWriter(log_dir="figures")

    bg_data, fg_data, sp_data, sf_data, score = load_data(config['debug'])
    # test_data = load_data(config['test_path'])

    Batcher = BatchGenerator(config, bg_data, fg_data, sp_data, sf_data, score)
    print("Batch number: ", Batcher.total)

    if config['Attention']:
        print("Use Attention")
        model = Verifier(config)
    else:
        model = VerifierBase(config)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 betas=(config['b1'], config['b2']),
                                 eps=config['e'],
                                 weight_decay=config['decay'])
    if config['Regression']:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss(reduce=False)

    if config['cuda']:
        model.cuda()
    model.train()

    for epc in range(config['epochs']):
        Batcher.reset()
        total_acc1 = 0
        total_acc2 = 0
        for i, batch in tqdm(enumerate(Batcher), total=len(Batcher)):
            global_step += 1
            # Just for test
            y1_pred, y2_pred = model(batch)
            if config['Regression']:
                loss1 = torch.sum(criterion(y1_pred[:, 0], batch['y1'].float()))
                loss2 = torch.sum(criterion(y2_pred[:, 0], batch['y2'].float()))
            else:
                loss1 = torch.sum(criterion(y1_pred, batch['y1']))
                loss2 = torch.sum(criterion(y2_pred, batch['y2']))
            loss = loss1 + config['lambda'] * loss2
            loss.backward()

            y1_pred, y2_pred = y1_pred.detach().cpu().numpy(), y2_pred.detach().cpu().numpy()

            print(loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy(), loss.detach().cpu().numpy())
            if i % 100 == 0:
                print(y1_pred, y2_pred)
                add_figure(config['name'], writer, global_step, loss1, loss2, loss)
                print(loss.detach().cpu().numpy())
            if global_step % 1000 == 0:
                acc, rmse = predict(Batcher, model)
                add_result(config['name'], writer, global_step, acc, rmse)
                print("acc: ", acc, "rmse: ", rmse)
            optimizer.step()
            optimizer.zero_grad()
        torch.save(model.state_dict(), "checkpoints/ckpt_{}_epoch_{}.pth".format(epc, config['name']))
    writer.close()
