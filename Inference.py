import random
from model.verifier_base import VerifierBase
import torch
from config import set_args
from torch.autograd import Variable
from utils.util import *
import time
import argparse

"""
Description:

    This is a script of Gradient-based Foreground Adjustment Algorithm.
    (x, y, Scale) of foreground objects will be adjust guided by model's gradient.
"""

# ========================== Constants =====================
parser = argparse.ArgumentParser(description='Inference Phase')
time = time.gmtime()
time = "-".join([str(p) for p in list(time)[:5]])
config = set_args()
test_fg = []

SAMPLE_NUM = config['sample_num']
ROUND = config['update_rd']
TOPK = config['top_k']

start_x = 0
start_y = 0
fx = [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1],
      [-1, 0, 0.95], [1, 0, 0.95], [0, -1, 0.95], [0, 1, 0.95],
      [-1, 0, 1.05], [1, 0, 1.05], [0, -1, 1.05], [0, 1, 1.05]]

# ======================== loading ckpt ================== #
ckpt = os.path.join("checkpoints", "ckpt_2_epoch_1:2:1_Regression_sigmoid_shuffle_score_debug.pth")
scene_parsing_folder_name = 'background_gallery_sp'
model_pred = VerifierBase(config)
#model_pred = Verifier(config)
model_pred.cuda()
model_pred.load_state_dict(torch.load(ckpt))
model_pred.eval()


def patch(v):
    v = Variable(v.cuda())
    return v


def f(background, foreground, scene_parsing):
    # -- TODO -- #
    colors = loadmat('resource/color150.mat')['colors']
    scene_parsing = colorEncode(scene_parsing, colors)
    batch = dict()
    batch['BGD'] = patch(torch.FloatTensor(background[:, :, :3].copy().transpose(2, 0, 1)).unsqueeze(0))
    batch['FGD'] = patch(torch.FloatTensor(foreground[:, :, :3].copy().transpose(2, 0, 1)).unsqueeze(0))
    batch['SPS'] = patch(torch.FloatTensor(scene_parsing[:, :, :3].copy().transpose(2, 0, 1)).unsqueeze(0))

    y1_pred, y2_pred = model_pred(batch)
    picture_match_score = y1_pred.detach().cpu().numpy()[..., 0]
    location_match_score = y2_pred.detach().cpu().numpy()[..., 0]
    print(picture_match_score[0], location_match_score[0])
    return [picture_match_score[0], location_match_score[0]]


def cvt2RGBA(img):
    _, _, channel = img.shape
    if channel == 4:
          return img
    if channel == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)


black_canvas = np.zeros((256, 256, 3), np.uint8)
black_canvas = np.concatenate([black_canvas, np.ones((256,256,1))*255.0], axis=2)


def paste(target, source, pos=(0,0)):
    left_up_x, left_up_y = pos
    bg_height, bg_width = target[:, :, 0].shape
    fg_height, fg_width = source[:, :, 0].shape
    result = target.copy()
    target_x_start = max(left_up_x, 0)
    target_x_end = min(left_up_x + fg_height, bg_height)
    target_y_start = max(left_up_y, 0)
    target_y_end = min(left_up_y + fg_width, bg_width)
    source_x_start = max(0, -left_up_x)
    source_x_end = min(bg_height-left_up_x, fg_height)
    source_y_start = max(0, -left_up_y)
    source_y_end = min(bg_width-left_up_y, fg_width)
    fg = source[source_x_start:source_x_end, source_y_start:source_y_end, :]
    bg = result[target_x_start:target_x_end, target_y_start:target_y_end, :]
    mask = fg[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
    fg = cv2.bitwise_and(fg, fg, mask=mask)
    result[target_x_start:target_x_end,
           target_y_start:target_y_end, :] = cv2.add(fg, bg)
    return result


def change(source, delta_x, delta_y, slope):
    alpha = source[:, :, 3]
    fg_height, fg_width = alpha.shape
    x0 = 0
    y0 = 0
    x1 = fg_height
    y1 = fg_width
    for i in range(fg_height):
        if np.sum(alpha[i, :]) != 0:
            x0 = i
            break
    for i in range(fg_height, x0, -1):
        if np.sum(alpha[i - 1, :]) != 0:
            x1 = i
            break
    for i in range(fg_width):
        if np.sum(alpha[:, i]) != 0:
            y0 = i
            break
    for i in range(fg_width, y0, -1):
        if np.sum(alpha[:, i - 1]) != 0:
            y1 = i
            break
    fg = source[x0:x1, y0:y1, :]
    new_fg = cv2.resize(fg, None, fx=slope, fy=slope)
    result = np.zeros(source.shape, np.uint8)
    result = paste(result, new_fg, (int(delta_x), int(delta_y)))
    return result


fg = cv2.imread(config['test_img'], -1)
fg = cvt2RGBA(fg)

rootpath = "/newNAS/Share/ykli"  # os.getcwd()
gallery_dir = "background_gallery"
os.mkdir(f'result/{time}')
picture_list = os.listdir(f'{rootpath}/{gallery_dir}/')
choosen_pictures = random.sample(picture_list, SAMPLE_NUM)

pic_scores = []
for picture_name in choosen_pictures:
    bg = cv2.imread(f'{rootpath}/{gallery_dir}/{picture_name}', -1)
    bg = cvt2RGBA(bg)
    with open(f'{rootpath}/{scene_parsing_folder_name}/{picture_name[0:-4]}.sg.pkl', 'rb') as fr:
          sp = pickle.load(fr)
    # try_pic = paste(bg, fg, start_x, start_y)
    # pic_scores.append(f(bg, fg, sp)[0])
    sc = f(bg, fg, sp)
    pic_scores.append(sc[0])

sorted_pic_scores = sorted(pic_scores)
# print(sorted_pic_scores)
theshold_score = sorted_pic_scores[TOPK - 1]
theshold_score_2 = sorted_pic_scores[SAMPLE_NUM - TOPK]
to_test_pictures = []
to_diss_pictures = []
for i in range(SAMPLE_NUM):
    if pic_scores[i] <= theshold_score:
        to_test_pictures.append(i)
    if pic_scores[i] >= theshold_score_2:
        to_diss_pictures.append(i)


# BAD CASES
for i_pic in range(TOPK):
    print("ipc", i_pic)
    picture_name = choosen_pictures[to_test_pictures[i_pic]]
    picture_score = pic_scores[to_test_pictures[i_pic]]

    os.mkdir(f'result/{time}/{picture_score}_{picture_name[0:-4]}')
    bg = cv2.imread(f'{rootpath}/{gallery_dir}/{picture_name}', -1)
    bg = cvt2RGBA(bg)
    bg_height, bg_width = bg[:, :, 0].shape
    mv_height = bg_height / 20
    mv_width = bg_width / 20
    with open(f'{rootpath}/{scene_parsing_folder_name}/{picture_name[0:-4]}.sg.pkl', 'rb') as fr:
          sp = pickle.load(fr)
    current_x = start_x
    current_y = start_y
    current_s = 1
    for iter_g in range(ROUND):
        tmp_pic_scores = []
        for i_fx in range(12):
            tmp_fg = change(fg, current_x + fx[i_fx][0]*mv_height, current_y + fx[i_fx][1]*mv_width, current_s * fx[i_fx][2])
            # try_pics = paste(bg, tmp_fg, start_x, start_y)
            tmp_pic_scores.append(f(bg, tmp_fg, sp)[1])
        max_index = tmp_pic_scores.index(max(tmp_pic_scores))
        current_x += fx[max_index][0]*mv_height
        current_y += fx[max_index][1]*mv_width
        current_s *= fx[max_index][2]
        mid_fg = change(fg, current_x, current_y, current_s)
        mid_result = paste(bg, mid_fg, (start_x, start_y))
        max_score = max(tmp_pic_scores)
        cv2.imwrite(f'./result/{time}/{picture_score}_{picture_name[0:-4]}/{iter_g}_{max_score}_{fx[max_index][0]}_{fx[max_index][1]}_{fx[max_index][2]}.png', mid_result)
    # final_fg = change(fg, current_x, current_y, current_s)
    # result = paste(bg, final_fg, start_x, start_y)
    # cv2.imwrite(f'./{i_pic}_{max_score}.png', result)


for i_pic in range(TOPK):
    print("ipc", i_pic)
    picture_name = choosen_pictures[to_diss_pictures[i_pic]]
    picture_score = pic_scores[to_diss_pictures[i_pic]]
    os.mkdir(f'result/{time}/{picture_score}_{picture_name[0:-4]}')
    bg = cv2.imread(f'{rootpath}/{gallery_dir}/{picture_name}', -1)
    bg = cvt2RGBA(bg)
    bg_height, bg_width = bg[:, :, 0].shape
    mv_height = bg_height / 20
    mv_width = bg_width / 20
    with open(f'{rootpath}/{scene_parsing_folder_name}/{picture_name[0:-4]}.sg.pkl', 'rb') as fr:
          sp = pickle.load(fr)
    current_x = start_x
    current_y = start_y
    current_s = 1
    for iter_g in range(ROUND):
        tmp_pic_scores = []
        for i_fx in range(12):
            tmp_fg = change(fg, current_x + fx[i_fx][0]*mv_height, current_y + fx[i_fx][1]*mv_width, current_s * fx[i_fx][2])
            # try_pics = paste(bg, tmp_fg, start_x, start_y)
            tmp_pic_scores.append(f(bg, tmp_fg, sp)[1])
        max_index = tmp_pic_scores.index(max(tmp_pic_scores))
        current_x += fx[max_index][0]*mv_height
        current_y += fx[max_index][1]*mv_width
        current_s *= fx[max_index][2]
        mid_fg = change(fg, current_x, current_y, current_s)
        mid_result = paste(bg, mid_fg, (start_x, start_y))
        max_score = max(tmp_pic_scores)
        cv2.imwrite(f'./result/{time}/{picture_score}_{picture_name[0:-4]}/{iter_g}_{max_score}_{fx[max_index][0]}_{fx[max_index][1]}_{fx[max_index][2]}.png', mid_result)
