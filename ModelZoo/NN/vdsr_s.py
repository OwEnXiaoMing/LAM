import torch
import torch.nn as nn
from math import sqrt

def make_model(args, parent=False):
    num_params = sum(param.numel() for param in (VDSR(args).parameters()))
    print(num_params)
    print('This is VDSR Shift')
    return VDSR(args)

class Shift_Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Shift_Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

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
        return self.relu(self.conv(self.shift_bi_features(x, 2, 4)))


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Shift_Conv_ReLU_Block, Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.scale = 4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, shift_block, block, num_of_layer):
        layers = []
        for i in range(num_of_layer):
            if i % 2 == 0:
                layers.append(shift_block())
            else:
                layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        residual = x1
        out = self.relu(self.input(x1))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

