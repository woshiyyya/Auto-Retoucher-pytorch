import cv2
import os
from os.path import expanduser
from os.path import join
from tqdm import tqdm
import pickle
import numpy as np
from scipy.io import loadmat


def shuffle_fn(idx, num):
    return '{}.shuffle.{}.jpg'.format(idx, num)


def accuracy(batch, y1_pred, y2_pred):
    y1 = batch['y1'].detach().cpu().numpy()
    y2 = batch['y2'].detach().cpu().numpy()
    # print(np.vstack([y1, y1_pred[:, 0]]))
    # print(np.vstack([y2, y2_pred[:, 0]]))
    y1_pred = np.argmax(y1_pred, axis=1)
    y2_pred = np.argmax(y2_pred, axis=1)
    acc1 = np.sum(y1 == y1_pred)/y1_pred.shape[0]
    acc2 = np.sum(y2 == y2_pred)/y2_pred.shape[0]
    # print(np.vstack([y1, y1_pred]))
    # print(np.vstack([y2, y2_pred]))
    return acc1, acc2


def add_figure(name, writer, global_step, loss1, loss2, loss):
    writer.add_scalar(name + ' data/train_loss', loss, global_step)
    writer.add_scalars(name + ' data/loss_group', {'loss1': loss1, 'loss2': loss2}, global_step)
    return


def add_result(name, writer, global_step, acc, rmse):
    writer.add_scalars(name + ' data/result_group', {'acc': acc, 'rmse': rmse}, global_step)
    return


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def load_data(small=False):
    source_path = os.path.join(expanduser("~"), "cvdata", "data_jpg")
    shuffle_path = os.path.join(expanduser("~"), "cvdata", "sigmoid_shuffle")
    scene_path = os.path.join(expanduser("~"), "cvdata", "scene_parsing_np")
    folder_names = ['golf', 'kitchen', 'office', 'airport_terminal', 'banquet',
                    'beach', 'boat', 'coffee_shop', 'conference_room', 'desert',
                    'football', 'hospital', 'ice_skating', 'stage', 'staircase',
                    'supermarket']
    folder_names = ['golf', 'kitchen', 'office', 'airport_terminal', 'banquet',
                    'beach', 'boat', 'coffee_shop', 'conference_room', 'desert',
                    'hospital', 'ice_skating', 'staircase',
                    'supermarket']
    if small:
        folder_names = ['desert']
    bg_data = dict()
    fg_data = dict()
    sp_data = dict()
    sf_data = dict()
    colors = loadmat('resource/color150.mat')['colors']

    for folder in folder_names:
        for img in tqdm(os.listdir(join(source_path, folder))):
            fn = join(source_path, folder, img)
            idx = img.split(".")[0]
            if "bg.jpg" in fn:
                bg_data[idx] = cv2.imread(fn).transpose(2, 0, 1)
            else:
                fg_data[idx] = cv2.imread(fn).transpose(2, 0, 1)

    for folder in folder_names:
        for img in tqdm(os.listdir(join(scene_path, folder))):
            fn = join(scene_path, folder, img)
            idx = img.split(".")[0]
            labels = pickle.load(open(fn, 'rb'))
            sp_data[idx] = colorEncode(labels, colors).transpose(2, 0, 1)

    for folder in folder_names:
        for img in tqdm(os.listdir(join(shuffle_path, folder))):
            fn = join(shuffle_path, folder, img)
            idx = img.split(".")[0]
            if idx in sf_data.keys():
                sf_data[idx].append(cv2.imread(fn).transpose(2, 0, 1))
            else:
                sf_data[idx] = [cv2.imread(fn).transpose(2, 0, 1)]

    score = pickle.load(open("/newNAS/Share/ykli/sigmoid_shuffle_score_dict.pkl", 'rb'))

    print(len(bg_data), list(bg_data.values())[0].shape)
    print(len(fg_data))
    print(len(sp_data), list(sp_data.values())[0].shape)
    print(len(sf_data))
    return bg_data, fg_data, sp_data, sf_data, score

