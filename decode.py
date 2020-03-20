"""
This script is to enhance the audio data using the trained model
Date: 2019/06
Author: Andong Li
"""
import torch
import argparse
import librosa
import os
import numpy as np
import json
import scipy
from Backup import *
import pickle
from data import *
from RTNet_RNN import  *
from RTNet_GRU import  *
from RTNet_LSTM import  *
from AECNN import  *
from RHR import  *


def enhance(args):
    model = RTNet_GRU.load_model(args.Model_path)
    print(model)
    model.eval()
    model.cuda()

    # Load data
    te_dataset = TestDataset(args.json_dir,
                               args.batch_size,
                             seen_flag = args.seen_flag)
    te_loader = TestDataLoader(te_dataset,
                               batch_size= 1,
                               num_workers= 0)
    with torch.no_grad():
        for (batch_id, batch_info) in enumerate(te_loader.get_data_loader()):
            recover_audio(batch_info, model, args= args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--recover_space', type = str, default= '/media/liandong/PROJECTS/RTNet/RECOVER',
                        help = 'The place to recover the utterances')
    parser.add_argument('--fs', type=int, default = 16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type = str, default = '/media/liandong/PROJECTS/RTNet/BEST_MODEL/rtnet_gru_stage_1_final_pth.tar',
                        help = 'The place to save best model')
    parser.add_argument('--json_dir', type=str, default='/media/Dataset/STIMIT/json',
                        help='The place to list the feat mat file and frame number')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='The number of the batch size')
    parser.add_argument('--seen_flag', type=int, default = 0,
                        help='if 1=> seen noise condition; 0 => unseen noise condition')
    args = parser.parse_args()
    print(args)
    enhance(args = args)