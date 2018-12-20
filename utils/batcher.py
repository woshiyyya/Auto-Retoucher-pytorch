import cv2
from os.path import join
import numpy as np
import torch
from torch.autograd import Variable
from utils.util import shuffle_fn
import random


class BatchGenerator(object):
    def __init__(self, config, bg_data, fg_data, sp_data, sf_data, score, is_training=True, ratio=[1, 2, 1]):
        self.config = config
        self.batch_size = config['batch_size']
        self.bg_data = bg_data
        self.fg_data = fg_data
        self.sp_data = sp_data
        self.sf_data = sf_data
        self.score = score
        self.keys = list(bg_data.keys())
        self.is_training = is_training
        self.ratio = ratio
        print("load data ok")

        self.offset = 0
        self.data = self.construct_dataset()
        self.total = len(self.data)
        self.clip_tail()
        self.batches = [self.data[i:i + self.batch_size] for i in range(0, self.total, self.batch_size)]
        if is_training:
            self.test_batches = self.batches[-int(0.2 * len(self.batches)):]
            self.batches = self.batches[:-int(0.2 * len(self.batches))]
        print("cut batch ok")

    @staticmethod
    def shuffle_key(data):
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        return [data[idx] for idx in indices]

    def construct_dataset(self):
        pos_bg = self.keys
        pos_fg = self.keys
        pos_fgid = [-1 for _ in range(len(pos_fg))]
        pos_y1 = [True for _ in range(len(pos_fg))]
        pos_y2 = [True for _ in range(len(pos_fg))]

        neg_sem_bg = self.keys
        neg_sem_fg = self.shuffle_key(self.keys)
        neg_sem_fgid = [-1 for _ in range(len(neg_sem_fg))]
        neg_sem_y1 = [False for _ in range(len(neg_sem_fg))]
        neg_sem_y2 = [False for _ in range(len(neg_sem_fg))]

        neg_spa_bg = []
        neg_spa_fgid = []
        neg_spa_y2 = []
        for key in self.keys:
            num_neg = self.ratio[1]
            neg_spa_bg.extend([key] * num_neg)
            idxs = [np.random.randint(0, 4) for _ in range(num_neg)]
            neg_spa_fgid.extend(idxs)
            neg_spa_y2.extend([self.score[shuffle_fn(key, idx)] for idx in idxs])
        neg_spa_fg = neg_spa_bg.copy()
        neg_spa_y1 = [True for _ in range(len(neg_spa_fg))]
        # neg_spa_y2 = [False for _ in range(len(neg_spa_fg))]
        # print(neg_spa_y2)

        # input()
        bg_keys = pos_bg * self.ratio[0] + neg_sem_bg * self.ratio[2] + neg_spa_bg
        fg_keys = pos_fg * self.ratio[0] + neg_sem_fg * self.ratio[2] + neg_spa_fg
        fg_ids = pos_fgid * self.ratio[0] + neg_sem_fgid * self.ratio[2] + neg_spa_fgid
        Y1 = pos_y1 * self.ratio[0] + neg_sem_y1 * self.ratio[2] + neg_spa_y1
        Y2 = pos_y2 * self.ratio[0] + neg_sem_y2 * self.ratio[2] + neg_spa_y2

        return self.shuffle_key([(bg, fg, fgid, y1, y2) for bg, fg, fgid, y1, y2 in zip(bg_keys, fg_keys, fg_ids, Y1, Y2)])

    def clip_tail(self):
        self.data = self.data[:self.total-(self.total % self.batch_size)]
        self.total = len(self.data)

    def reset(self):
        self.offset = 0
        if self.is_training:
            indices = list(range(self.total))
            np.random.shuffle(indices)
            self.data = [self.data[idx] for idx in indices]
        return

    def patch(self, v):
        if self.config['cuda']:
            v = Variable(v.cuda())
        else:
            v = Variable(v)
        return v

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.batches[self.offset]
            self.offset += 1

            # True Backgrounds <--> True Foregrounds [N]
            foregrounds = []
            for case in batch:
                if case[2] == -1:
                    foregrounds.append(self.fg_data[case[1]])
                else:
                    foregrounds.append(self.sf_data[case[1]][case[2]])
            backgrounds = [self.bg_data[case[0]] for case in batch]
            sceneparsing = [self.sp_data[case[0]] for case in batch]
            y1 = [case[3] for case in batch]
            y2 = [case[4] for case in batch]

            batch_dict = dict()
            batch_dict['BGD'] = self.patch(torch.FloatTensor(backgrounds))
            batch_dict['FGD'] = self.patch(torch.FloatTensor(foregrounds))
            batch_dict['SPS'] = self.patch(torch.FloatTensor(sceneparsing))
            batch_dict['y1'] = self.patch(torch.LongTensor(y1))
            batch_dict['y2'] = self.patch(torch.FloatTensor(y2))
            # print(batch_dict['BGD'].shape)
            # print(batch_dict['FGD'].shape)
            # print(batch_dict['SPS'].shape)
            # print(batch_dict['y1'].shape)
            # print(batch_dict['y2'].shape)

            yield batch_dict
        return

    def batch2cuda(self, batch):
        foregrounds = []
        for case in batch:
            if case[2] == -1:
                foregrounds.append(self.fg_data[case[1]])
            else:
                foregrounds.append(self.sf_data[case[1]][case[2]])
        backgrounds = [self.bg_data[case[0]] for case in batch]
        sceneparsing = [self.sp_data[case[0]] for case in batch]
        y1 = [case[3] for case in batch]
        y2 = [case[4] for case in batch]

        batch_dict = dict()
        batch_dict['BGD'] = self.patch(torch.FloatTensor(backgrounds))
        batch_dict['FGD'] = self.patch(torch.FloatTensor(foregrounds))
        batch_dict['SPS'] = self.patch(torch.FloatTensor(sceneparsing))
        batch_dict['y1'] = self.patch(torch.LongTensor(y1))
        batch_dict['y2'] = self.patch(torch.FloatTensor(y2))
        return batch_dict