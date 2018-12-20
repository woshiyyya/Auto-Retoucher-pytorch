import cv2
from os.path import join
import numpy as np
import torch
from torch.autograd import Variable
import random


class BatchGenerator(object):
    def __init__(self, config, bg_data, fg_data, sp_data, sf_data, is_training=True, ratio=[14, 6, 2]):
        self.config = config
        self.sample_size = ratio[0]  # config['sample_size']
        self.batch_size = ratio[0]  # config['sample_size'] * 6
        self.bg_data = bg_data
        self.fg_data = fg_data
        self.sp_data = sp_data
        self.sf_data = sf_data
        self.data = list(bg_data.keys())
        self.is_training = is_training
        self.ratio = ratio
        print("load data ok")
        self.total = len(self.data)

        self.offset = 0
        # self.set_number()
        print(self.total)
        indices = list(range(self.total))
        np.random.shuffle(indices)
        self.neg_data = [self.data[idx] for idx in indices]

        self.clip_tail()
        self.pos_batches = [self.data[i:i + self.sample_size] for i in range(0, self.total, self.sample_size)]
        self.neg_batches = [self.neg_data[i:i + self.sample_size] for i in range(0, self.total, self.sample_size)]
        print("cut batch ok")

    def set_number(self):
        self.color_neg_num = int(self.batch_size * (self.ratio[1] / sum(self.ratio)))
        self.position_neg_num = int(self.batch_size * (self.ratio[2] / sum(self.ratio)))
        self.pos_num = self.batch_size - self.color_neg_num - self.position_neg_num

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
        return len(self.pos_batches)

    def __iter__(self):
        while self.offset < len(self):
            pos_batch = self.pos_batches[self.offset]
            neg_batch = self.neg_batches[self.offset]
            self.offset += 1

            # True Backgrounds <--> True Foregrounds [N]
            pos_backgrounds = [self.bg_data[idx] for idx in pos_batch]
            pos_foregrounds = [self.fg_data[idx] for idx in pos_batch]
            pos_sceneparsing = [self.sp_data[idx] for idx in pos_batch]
            pos_y1 = [True for _ in range(self.sample_size)]
            pos_y2 = [True for _ in range(self.sample_size)]

            # True Backgrounds <--> False Foregrounds [N]
            neg_col_backgrounds = []
            neg_col_foregrounds = []
            neg_col_sceneparsing = []
            for i in range(self.ratio[1]):
                neg_col_backgrounds.append(self.bg_data[pos_batch[i]])
                neg_col_foregrounds.append(self.fg_data[neg_batch[i]])
                neg_col_sceneparsing.append(self.sp_data[pos_batch[i]])
            neg_col_y1 = [False for _ in range(self.ratio[1])]
            neg_col_y2 = [False for _ in range(self.ratio[1])]

            # True Backgrounds <--> True Foregrounds & False Position  [4N]
            neg_pos_backgrounds = []
            neg_pos_foregrounds = []
            neg_pos_sceneparsing = []
            neg_pos_y1 = [True for _ in range(self.ratio[2] * 4)]
            neg_pos_y2 = [False for _ in range(self.ratio[2] * 4)]
            for i in range(self.ratio[2]):
                idx = pos_batch[i]
                for k in range(4):
                    neg_pos_backgrounds.append(self.bg_data[idx])
                    neg_pos_foregrounds.append(self.sf_data[idx][k])
                    neg_pos_sceneparsing.append(self.sp_data[idx])

            BGD = pos_foregrounds + neg_col_foregrounds + neg_pos_foregrounds
            FGD = pos_backgrounds + neg_col_backgrounds + neg_pos_backgrounds
            SPS = pos_sceneparsing + neg_col_sceneparsing + neg_pos_sceneparsing
            y1 = pos_y1 + neg_col_y1 + neg_pos_y1
            y2 = pos_y2 + neg_col_y2 + neg_pos_y2

            # Shuffle Data
            N = len(y1)
            indices = list(range(N))
            random.shuffle(indices)
            BGD = [BGD[idx].tolist() for idx in indices]
            FGD = [FGD[idx].tolist() for idx in indices]
            SPS = [SPS[idx].tolist() for idx in indices]
            y1 = [y1[idx] for idx in indices]
            y2 = [y2[idx] for idx in indices]

            batch_dict = dict()
            batch_dict['BGD'] = self.patch(torch.FloatTensor(BGD))
            batch_dict['FGD'] = self.patch(torch.FloatTensor(FGD))
            batch_dict['SPS'] = self.patch(torch.FloatTensor(SPS))
            batch_dict['y1'] = self.patch(torch.LongTensor(y1))
            batch_dict['y2'] = self.patch(torch.LongTensor(y2))
            print(batch_dict['BGD'].shape)
            print(batch_dict['FGD'].shape)
            print(batch_dict['SPS'].shape)
            print(batch_dict['y1'].shape)
            print(batch_dict['y2'].shape)

            yield batch_dict

        return
