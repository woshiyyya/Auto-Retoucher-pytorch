from model import *
import torch
from utils.Batcher import BatchGenerator


def predict(ckpt, batch):
    model_pred = BackgroundMatcher()
    model_pred.cuda()
    model_pred.load_state_dict(torch.load(ckpt))
    model_pred.eval()
    y1_pred, y2_pred = model_pred(batch)
    y1_pred = y1_pred.detach().cpu().numpy()
    y2_pred = y2_pred.detach().cpu().numpy()
    print("y1:", y1_pred)
    print("y2:", y2_pred)
    return y1_pred, y2_pred


if __name__ == "__main__":

    predict()