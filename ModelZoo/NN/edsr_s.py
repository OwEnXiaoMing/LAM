from ..NN import common

import torch.nn as nn
import torch

print('this is edsr shift bi')

url = {

}




def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def report_model(model):
    n_parameters = 0
    for p in model.parameters():
        n_parameters += p.nelement()
    print("Model  with {} parameters".format(n_parameters))


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


import math


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResBlock_shift_bi(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_shift_bi, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def shift_quad_features(self, input, move, m_c=0):
        H = input.shape[2]
        W = input.shape[3]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left1 = torch.zeros_like(input[:, :m_c])
        zero_right1 = torch.zeros_like(input[:, :m_c])
        zero_left2 = torch.zeros_like(input[:, :m_c])
        zero_right2 = torch.zeros_like(input[:, :m_c])

        zero_left1[:, :, :-move, :] = input[:, mid_channel - m_c * 2:mid_channel - m_c, move:, :]  # up
        zero_left2[:, :, move:, :] = input[:, mid_channel - m_c:mid_channel, :H - move, :]  # down
        zero_right1[:, :, :, :-move] = input[:, mid_channel:mid_channel + m_c, :, move:]  # left
        zero_right2[:, :, :, move:] = input[:, mid_channel + m_c:mid_channel + m_c * 2, :, :W - move]  # right

        return torch.cat(
            (input[:, 0:mid_channel - m_c * 2], zero_left1, zero_left2, zero_right1, zero_right2,
            input[:, mid_channel + m_c * 2:]),
            1)

    def shift_bi_features(self, input, move, m_c=0):
        H = input.shape[2]
        channel_size = input.shape[1]
        mid_channel = channel_size // 2

        zero_left = torch.zeros_like(input[:, :m_c])
        zero_right = torch.zeros_like(input[:, :m_c])
        zero_left[:, :, :-move, :] = input[:, mid_channel - m_c:mid_channel, move:, :]
        zero_right[:, :, move:, :] = input[:, mid_channel:mid_channel + m_c, :H - move, :]
        return torch.cat((input[:, 0:mid_channel - m_c], zero_left, zero_right, input[:, mid_channel + m_c:]), 1)

    def forward(self, x):
        x1 = self.shift_bi_features(x, 6, 4)
        #x1 = self.shift_quad_features(x, 20, 4)
        res = self.body(x1).mul(self.res_scale)
        res += x
        return res


class EDSR_s(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDSR_s, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(3)
        self.add_mean = MeanShift(3, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock_shift_bi(
                conv, n_feats, kernel_size, act=act,
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
