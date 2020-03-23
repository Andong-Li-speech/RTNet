"""
This script includes AudioDataSet and AudioDataLoader, where AudioDataSet is a class to return miniatch list and
AudioDataLoader is to load the minibatch data from the list returned by AudioDataSet
This code is based on the code written by Kaituo Xu
Date: 2019/06
Author:Andong Li
"""

import json
import os
import h5py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import librosa
import random
import soundfile as sf
from Backup import *
from config import *


class To_Tensor(object):
    def __call__(self, x, type):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return  torch.IntTensor(x)


class SignalToFrames:
    def __init__(self, frame_size=2048, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        frame_size = self.frame_size
        frame_shift = self.frame_shift
        sig_len = len(in_sig)
        nframes = (sig_len // self.frame_shift)
        a = np.zeros([nframes, self.frame_size])
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[i, :] = in_sig[start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[i, :tail_size] = in_sig[start:]
            start = start + self.frame_shift
            end = start + self.frame_size
        return a

class TrainDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        feat_json_pos= os.path.join(json_dir, 'train', 'mix.json')
        label_json_pos= os.path.join(json_dir, 'train', 'clean.json')
        with open(feat_json_pos, 'r') as f:
            feat_json_list = json.load(f)
        with open(label_json_pos, 'r') as f:
            label_json_list = json.load(f)

        feat_minibatch, label_minibatch = [], []
        start = 0
        while True:
            end = min(len(feat_json_list), start+ batch_size)
            feat_minibatch.append(feat_json_list[start:end])
            label_minibatch.append(label_json_list[start:end])
            start = end
            if end == len(feat_json_list):
                break
        self.feat_minibatch = feat_minibatch
        self.label_minibatch = label_minibatch

    def __len__(self):
        return len(self.feat_minibatch)

    def __getitem__(self, index):
        return self.feat_minibatch[index], self.label_minibatch[index]


class CvDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        feat_json_pos = os.path.join(json_dir, 'cv', 'mix.json')
        label_json_pos = os.path.join(json_dir, 'cv', 'clean.json')
        with open(feat_json_pos, 'r') as f:
            feat_json_list = json.load(f)
        with open(label_json_pos, 'r') as f:
            label_json_list = json.load(f)

        feat_minibatch, label_minibatch = [], []
        start = 0
        while True:
            end = min(len(feat_json_list), start + batch_size)
            feat_minibatch.append(feat_json_list[start:end])
            label_minibatch.append(label_json_list[start:end])
            start = end
            if end == len(feat_json_list):
                break
        self.feat_minibatch = feat_minibatch
        self.label_minibatch = label_minibatch

    def __len__(self):
        return len(self.feat_minibatch)

    def __getitem__(self, index):
        return self.feat_minibatch[index], self.label_minibatch[index]


class TestDataset(Dataset):
    def __init__(self, json_dir, batch_size, seen_flag):
        self.json_dir = json_dir
        self.batch_size = batch_size
        if seen_flag == 1:
            feat_json_pos = os.path.join(json_dir, 'seen_test', 'mix.json')
            label_json_pos = os.path.join(json_dir, 'seen_test', 'clean.json')
        else:
            feat_json_pos = os.path.join(json_dir, 'unseen_test', 'mix.json')
            label_json_pos = os.path.join(json_dir, 'unseen_test', 'clean.json')

        with open(feat_json_pos, 'r') as f:
            feat_json_list = json.load(f)
        with open(label_json_pos, 'r') as f:
            label_json_list = json.load(f)

        feat_minibatch, label_minibatch = [], []
        start = 0
        while True:
            end = min(len(feat_json_list), start + batch_size)
            feat_minibatch.append(feat_json_list[start:end])
            label_minibatch.append(label_json_list[start:end])
            start = end
            if end == len(feat_json_list):
                break
        self.feat_minibatch = feat_minibatch
        self.label_minibatch = label_minibatch

    def __len__(self):
        return len(self.feat_minibatch)

    def __getitem__(self, index):
        return self.feat_minibatch[index], self.label_minibatch[index]


class TrainDataLoader(object):
    def __init__(self, data_set, batch_size, num_workers = 0):

        self.data_loader = DataLoader(dataset= data_set,
                                           batch_size = batch_size,
                                           shuffle = 1,
                                           num_workers= num_workers,
                                           collate_fn= self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_list)

    def get_data_loader(self):
        return self.data_loader

def generate_feats_labels(batch):
    feat_batch, label_batch = batch[0][0], batch[0][1]
    feat_list, label_list, frame_list = [], [], []
    to_tensor = To_Tensor()
    signal_to_frame = SignalToFrames(frame_size= win_size, frame_shift= win_shift)
    for id in range(len(feat_batch)):
        feat_wav, _= sf.read(feat_batch[id])
        label_wav, _ = sf.read(label_batch[id])
        ones = np.ones(feat_wav.shape).astype(np.int)

        if len(feat_wav) > nsamples:
            wav_start = random.randint(0, len(feat_wav)- nsamples)
            feat_wav = feat_wav[wav_start:wav_start+nsamples]
            label_wav = label_wav[wav_start:wav_start+nsamples]
            ones = ones[wav_start:wav_start+nframes]
        else:
            feat_wav = np.concatenate((feat_wav, np.zeros(nsamples - len(feat_wav))))
            ones = np.concatenate((ones, np.zeros(nframes - len(label_wav))))
            label_wav = np.concatenate((label_wav, np.zeros(nsamples - len(label_wav))))

        feat_x = signal_to_frame(feat_wav)
        label_x = signal_to_frame(label_wav)
        ones_x = signal_to_frame(ones)
        feat_list.append(feat_x)
        label_list.append(label_x)
        frame_list.append(ones_x)
    feat_list = to_tensor(np.concatenate(feat_list, axis= 0), 'float')
    label_list = to_tensor(np.concatenate(label_list, axis = 0), 'float')
    frame_list = to_tensor(np.concatenate(frame_list, axis = 0), 'float').requires_grad_(False)
    return feat_list.cuda(), label_list.cuda(), frame_list.cuda()

class CvDataLoader(object):
    def __init__(self, data_set, batch_size, num_workers = 0):

        self.data_loader = DataLoader(dataset= data_set,
                                           batch_size = batch_size,
                                           shuffle = 1,
                                           num_workers= num_workers,
                                           collate_fn= self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_list)

    def get_data_loader(self):
        return self.data_loader


class BatchInfo(object):
    def __init__(self, feats, labels, frame_list):
        self.feats = feats
        self.labels = labels
        self.frame_list = frame_list

class TestDataLoader(object):
    def __init__(self, data_set, batch_size, num_workers = 0):

        self.data_loader = DataLoader(dataset= data_set,
                                           batch_size = batch_size,
                                           shuffle = 1,
                                           num_workers= num_workers,
                                           collate_fn= self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_list, info_list = test_generate_feats_labels(batch)
        return Test_BatchInfo(feats, labels, frame_list, info_list)

    def get_data_loader(self):
        return self.data_loader

def test_generate_feats_labels(batch):
    feat_batch, label_batch = batch[0][0], batch[0][1]
    feat_list, label_list, frame_list, info_list = [], [], [], []
    signal_to_frame = SignalToFrames(frame_size= win_size, frame_shift= win_shift)
    to_tensor = To_Tensor()
    for id in range(len(feat_batch)):
        file_path = feat_batch[id]
        feat_wav, _= sf.read(feat_batch[id])
        label_wav, _ = sf.read(label_batch[id])
        feat_x = signal_to_frame(feat_wav)
        label_x = signal_to_frame(label_wav)
        feat_list.append(feat_x)
        label_list.append(label_x)
        frame_list.append(feat_x.shape[0])
        info_list.append(os.path.splitext(os.path.split(file_path)[1])[0])
    feat_list = to_tensor(np.concatenate(feat_list, axis= 0), 'float')
    return feat_list.cuda(), label_list, frame_list, info_list

class Test_BatchInfo(object):
    def __init__(self, feats, labels, frame_list, info_list):
        self.feats = feats
        self.labels = labels
        self.frame_list = frame_list
        self.info_list = info_list