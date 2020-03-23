"""
This script is the backup function used to support backup support for the SE system
Author: Andong Li
Time: 2019/06
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pickle
import json
import os
import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pystoi.stoi import stoi
from python_speech_features.sigproc import framesig, deframesig
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
from config import  *

EPSILON = 1e-10

def calc_sp(mix, clean, data_type, Win_Length, Offset_Length):
    """
    This func is to calculate the features and corresponding labels in the time domain
    :param mix:   1-D vector
    :param clean:  1-D vector
    :param data_type: 
    :param Win_Length: the length of the window function
    :param Offset_Length: the offset length between adjanct frames
    :return: 
    """"""
    """
    n_window= Win_Length
    n_offset = Offset_Length

    mix_x = framesig(mix,
                     frame_len= n_window,
                     frame_step = n_offset)
    clean_x = framesig(clean,
                       frame_len = n_window,
                       frame_step= n_offset)

    return mix_x, clean_x


def contexual_frame_add(data, n_concate, n_hop):
    """
    This func is to make contexual frame concatenation
    :param data: input feature matrix, size: frame x feature
    :param n_concate: =(2 * n_hop + 1)
    :param n_hop:
    :return:  n_concate x feature
    """
    data = pad_with_border(data,n_hop)
    frame_num, feature_num = data.shape
    out = []
    ct = 0
    while (ct+ n_concate <= frame_num):
        out.append(data[ct: ct + n_concate])
        ct += 1
    out = np.concatenate(out , axis = 0)
    out = np.reshape(out, (-1, n_concate, feature_num))
    out = np.reshape(out, (out.shape[0], -1))
    return  out

def pad_with_border(x,n_hop):
    x_pad_list = [x[0:1]] * n_hop + [x] + [x[-1:]] * n_hop
    x_pad_list = np.concatenate(x_pad_list, axis= 0)
    return x_pad_list

def mean_cal(global_scaler_mean, group_number_list):
    mean_value = 0
    for i in range(len(global_scaler_mean)):
        mean_value += global_scaler_mean[i] * group_number_list[i] / np.sum(np.array(group_number_list))
    return mean_value

def cal_local_std(x , global_mean , total_num):
    """
    This func is to calculate the loacal std using local dataset and global mean
    :param x:
    :param global_mean:
    :param frame_num:
    :return:
    """
    accu = 0
    for i in range(np.shape(x)[0]):
        accu += np.power(x[i,:]-global_mean,2)
    accu = accu / total_num
    return accu


def batch_cal_max_frame(file_infos):
    max_frame = 0
    for utter_infos in zip(file_infos):
        file_path = utter_infos[0]
        # read mat file
        mat_file = h5py.File(file_path[0])
        mix_feat = np.transpose(mat_file['mix_feat'])
        max_frame = np.max([max_frame, mix_feat.shape[0]])
    return max_frame

def de_pad(pack):
    """
    clear the zero value in each batch tensor
    Note: return is a numpy format instead of Tensor
    :return:
    """
    mix = pack.mix[0:pack.frame_list,:]
    esti = pack.esti[0:pack.frame_list,:]
    speech = pack.speech[0:pack.frame_list,:]
    return mix, esti, speech


class decode_pack(object):
    def __init__(self, mix, esti, speech, frame_list):
        self.mix = mix
        self.esti = esti
        self.speech = speech
        self.frame_list = frame_list.astype(np.int32)


def ola(inputs, win_size, win_shift):
    nframes = inputs.shape[-2]
    sig_len = (nframes - 1)* win_shift + win_size
    sig = np.zeros((sig_len,))
    ones = np.zeros((sig.shape))
    start = 0
    end = start + win_size
    for i in range(nframes):
        sig[start:end] += inputs[i, :]
        ones[start:end] += 1
        start = start + win_shift
        end= start + win_size
    return sig / ones


def recover_audio(batch_info, model, args):
    """
    This func is to recover the audio by iSTFT and overlap-add
    :param pack:  pack is a class, consisting of four components
    :param args:
    :return:
    """
    _, esti_list = model(batch_info.feats)
    esti = esti_list[-3].squeeze().cpu().numpy()
    mix = batch_info.feats.cpu().numpy()
    speech_list = batch_info.labels
    frame_list = batch_info.frame_list
    info_list = batch_info.info_list

    # The path to write audio
    if args.seen_flag == 1:
        write_out_dir = os.path.join(args.recover_space, 'seen recover')
    else:
        write_out_dir = os.path.join(args.recover_space, 'unseen recover')
    os.makedirs(write_out_dir, exist_ok=True)
    clean_write_dir = os.path.join(write_out_dir, 'clean')
    os.makedirs(clean_write_dir, exist_ok=True)
    esti_write_dir = os.path.join(write_out_dir, 'esti')
    os.makedirs(esti_write_dir, exist_ok=True)
    mix_write_dir = os.path.join(write_out_dir, 'mix')
    os.makedirs(mix_write_dir, exist_ok=True)

    cnt = 0
    for i in range(len(frame_list)):
        de_mix = mix[cnt:cnt+ frame_list[i]]
        de_speech = speech_list[i]
        de_esti = esti[cnt:cnt+ frame_list[i]]
        cnt += frame_list[i]

        mix_utt = deframesig(de_mix, siglen= 0, frame_len= win_size, frame_step = win_shift).astype(np.float32)
        esti_utt = deframesig(de_esti, siglen= 0, frame_len= win_size, frame_step = win_shift).astype(np.float32)
        clean_utt = deframesig(de_speech, siglen= 0, frame_len= win_size, frame_step= win_shift).astype(np.float32)
        #mix_utt = ola(de_mix, win_size= win_size, win_shift= win_shift)
        esti_utt = ola(de_esti, win_size=win_size, win_shift=win_shift)
        #clean_utt = ola(de_speech, win_size=win_size, win_shift=win_shift)

        filename= os.path.split(info_list[i])[1]
        os.makedirs(os.path.join(esti_write_dir), exist_ok=True)
        os.makedirs(os.path.join(mix_write_dir), exist_ok=True)
        os.makedirs(os.path.join(clean_write_dir), exist_ok=True)
        librosa.output.write_wav(os.path.join(esti_write_dir, '%s_enhanced.wav' % filename),
                                 esti_utt, args.fs)
        # librosa.output.write_wav(os.path.join(clean_write_dir, '%s_clean.wav' % filename), clean_utt,
        #                          args.fs)
        # librosa.output.write_wav(os.path.join(mix_write_dir, '%s_mix.wav' % filename), mix_utt,
        #                          args.fs)




def mae_loss(esti, label, frames):

    esti  = torch.squeeze(esti)
    esti = esti * frames
    label = label * frames
    loss = torch.abs(torch.squeeze(esti - label)).mean()
    return loss


def sdr_loss(esti, label, granularity):
    esti = torch.squeeze(esti)
    block = np.int(np.ceil(esti.size(1) / granularity))
    cnt = 0
    sdr_lo = torch.zeros((esti.size(0))).cuda()
    c1 = 10
    for i in range(block):
        es = esti[:, cnt: np.min((cnt + granularity, esti.size(1)))]
        la = label[:, cnt: np.min((cnt + granularity, esti.size(1)))]
        sdr_lo = sdr_lo + c1 * (1 - F.cosine_similarity(es, la, dim = 1)) / 2
        cnt = cnt + granularity
    return (sdr_lo / block).mean() + EPSILON




def summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file)
    return count
















