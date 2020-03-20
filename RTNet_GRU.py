"""
This script is the place to save various networks
Date: 2019.06
Author: Andong Li
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class RTNet_GRU(nn.Module):
    def __init__(self):
        super(RTNet_GRU, self).__init__()
        # Main Encoder Part
        self.gru = Stage_GRU()
        self.en = Encoder()
        self.de = Decoder()
        self.glu_list = nn.ModuleList([GLU(dila_rate= 2**i) for i in range(6)])
        # Iteration Num
        self.Iter = 1

    def forward(self, mixture):
        """
        :param mixture: [B, T, F] B: Batch; T: Timestep
                            F: Feature;
        :return:
        """
        mixture = mixture.unsqueeze(dim= 1)
        x = mixture
        batch_size, feat_dim = mixture.size(0), mixture.size(2)
        h = Variable(torch.zeros(batch_size, 16, 1024)).to(x.device)
        x_list = []
        for i in range(self.Iter):
            x = torch.cat((mixture, x), dim= 1)
            h = self.gru(x, h)
            x, en_list = self.en(h)
            skip = Variable(torch.zeros(x.shape), requires_grad = True).cuda()
            for id in range(6):
                x = self.glu_list[id](x)
                skip = skip + x
            x = skip
            x = self.de(x, en_list)
            x_list.append(x)
        return x, x_list

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location= lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch ,tr_loss = None, cv_loss = None):
        package= {
            'gru':model.gru,
            'en': model.en,
            'glu_list': model.glu_list,
            'de': model.de,
            'Iter': model.Iter,
            'state_dict':model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.en1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.PReLU(16))           # 1024x16
        self.en2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2, padding=5),
            nn.PReLU(32))           # 512x32
        self.en3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=2, padding=5),
            nn.PReLU(64))   # 256x64
        self.en4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=2, padding=5),
            nn.PReLU(128))    # 128x128

    def forward(self, x):
        en_list = []
        x = self.en1(x)
        en_list.append(x)
        x = self.en2(x)
        en_list.append(x)
        x = self.en3(x)
        en_list.append(x)
        x = self.en4(x)
        en_list.append(x)
        return x, en_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.de4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128 + 128, out_channels=64, kernel_size=11, stride=2, padding=5,
                               output_padding=1),
            nn.PReLU(64))    # 64x256
        self.de3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels= 64 + 64, out_channels=32, kernel_size=11, stride=2, padding=5,
                               output_padding=1),
            nn.PReLU(32))      # 32x512
        self.de2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32 + 32, out_channels=16, kernel_size=11, stride=2, padding=5,
                               output_padding=1),  # 16 x 1024
            nn.PReLU(16))
        self.de1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16 + 16, out_channels=1, kernel_size=11, stride=2, padding=5,
                               output_padding=1),
            nn.Tanh())
    def forward(self, x, x_list):
        x = self.de4(torch.cat((x, x_list[-1]), dim= 1))
        x = self.de3(torch.cat((x, x_list[-2]), dim = 1))
        x = self.de2(torch.cat((x, x_list[-3]), dim= 1))
        x = self.de1(torch.cat((x, x_list[-4]), dim= 1))
        return x



class Stage_GRU(nn.Module):
    def __init__(self):
        super(Stage_GRU, self).__init__()
        # Recurrent Part
        # Recurrent Part
        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=11, stride=2, padding=5),
            nn.PReLU(16))  # 1024x 16
        self.conv_z = nn.Sequential(
            nn.Conv1d(in_channels=16 + 16, out_channels=16, kernel_size=11, stride=1, padding=5),  # 1024x16
            nn.Sigmoid())
        self.conv_r = nn.Sequential(
            nn.Conv1d(in_channels=16 + 16, out_channels=16, kernel_size=11, stride=1, padding=5),  # 1024x16
            nn.Sigmoid())
        self.conv_n = nn.Sequential(
            nn.Conv1d(in_channels=16 + 16, out_channels=16, kernel_size=11, stride=1, padding=5),  # 1024x16
            nn.Tanh())

    def forward(self, x, h= None):
        x = self.pre_conv(x)
        x1 = x
        x = torch.cat((x, h), dim = 1)
        z = self.conv_z(x)
        r = self.conv_r(x)
        s = r * h
        s = torch.cat((s, x1), dim =1)
        n = self.conv_n(s)
        h = (1- z) * h + z * n
        return h

class GLU(nn.Module):
    def __init__(self, dila_rate):
        super(GLU, self).__init__()
        self.in_conv = nn.Conv1d(in_channels = 128, out_channels= 64, kernel_size= 1, stride = 1)
        self.dila_conv_left = nn.Sequential(
            nn.PReLU(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 11, stride = 1,
                                   padding= np.int((dila_rate * 10) / 2), dilation= dila_rate))
        self.dila_conv_right = nn.Sequential(
            nn.PReLU(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 11, stride = 1,
                                   padding= np.int((dila_rate * 10) / 2), dilation= dila_rate),
            nn.Sigmoid())
        self.out_conv = nn.Conv1d(in_channels= 64, out_channels= 128, kernel_size = 1, stride = 1)
        self.out_prelu = nn.PReLU(128)

    def forward(self, inpt):
        x = inpt
        x = self.in_conv(x)
        x1 = self.dila_conv_left(x)
        x2 = self.dila_conv_right(x)
        x = x1 * x2
        x = self.out_conv(x)
        x = x + inpt
        x = self.out_prelu(x)
        return x