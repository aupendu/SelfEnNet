import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def make_model(args, parent=False):
    return SelfEnNet(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel, bias):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class CRB(nn.Module):
    def __init__(self, dim, bias):
        super(CRB, self).__init__()
        self.inp1 = default_conv(dim, dim, 3, bias) 
        self.inp2 = default_conv(dim, dim, 3, bias) 
        self.relu = nn.ReLU(inplace=True)
        self.calayer = CALayer(dim, bias)

    def forward(self, x):

        inp = self.relu(self.inp1(x))
        inp = self.relu(self.inp2(inp))
        inp = self.calayer(inp)

        return x + inp

class Gamma_Estimation(nn.Module):
    def __init__(self):
        super(Gamma_Estimation, self).__init__()

        feat = 32
        self.conv_input = nn.Conv2d(3, feat, 3, 1, 1, bias=False) 
        self.conv_cond  = nn.Conv2d(1, feat, 3, 1, 1, bias=False) 
        self.conv_mix = nn.Conv2d(2*feat, feat, 3, 1, 1, bias=False) 
        self.relu = nn.ReLU(True)
        self.CRB_blocks = nn.Sequential(
            CRB(feat, False),
            CRB(feat, False),
            CRB(feat, False),
            CRB(feat, False)
        )
        self.conv_out  = nn.Conv2d(feat, 1, 3, 1, 1, bias=False) 

    def forward(self, low_input, est_enhance):
        cond = self.relu(self.conv_cond(est_enhance))
        feat  = self.relu(self.conv_input(low_input))

        feat = self.relu(self.conv_mix(torch.cat([feat, cond], 1)))

        r = self.CRB_blocks(feat)
        r = self.relu(self.conv_out(r))
        return r

class Denoise_Network(nn.Module):
    def __init__(self):
        super(Denoise_Network, self).__init__()

        feat = 32
        self.conv_input = nn.Conv2d(3, feat, 3, 1, 1, bias=False) 
        self.conv_cond  = nn.Conv2d(1, feat, 3, 1, 1, bias=False) 
        self.conv_mix = nn.Conv2d(2*feat, feat, 3, 1, 1, bias=False) 
        self.relu = nn.ReLU(True)
        self.CRB_blocks = nn.Sequential(
            CRB(feat, False),
            CRB(feat, False),
            CRB(feat, False),
            CRB(feat, False)
        )
        self.conv_out  = nn.Conv2d(feat, 3, 3, 1, 1, bias=False) 

    def forward(self, low_input, r):

        cond = self.relu(self.conv_cond(r))
        feat  = self.relu(self.conv_input(low_input))

        feat = self.relu(self.conv_mix(torch.cat([feat, cond], 1)))


        out = self.CRB_blocks(feat)
        out = self.relu(self.conv_out(out))
        return out


class SelfEnNet(nn.Module):
    def __init__(self, args):
        super(SelfEnNet, self).__init__()

        if args.level>0: self.noise  = Denoise_Network()

        self.gamma = Gamma_Estimation()
        self.fac = nn.Parameter(torch.FloatTensor([1/2.2]), requires_grad=False)
        self.itr = args.iterations
        self.args = args

    def forward(self, x):
        xlow, num = x

        if num=='enhance_0':
            r = self.fac.expand(xlow.shape[0], 1, xlow.shape[2], xlow.shape[3])
            for _ in range(self.itr):
                r = self.gamma(xlow, r)
            est_high = self.enhance(xlow, r)
            return est_high, r, 0

        if num=='denoise_1':
            r = self.fac.expand(xlow.shape[0], 1, xlow.shape[2], xlow.shape[3])
            with torch.no_grad():
                for _ in range(self.itr):
                    r = self.gamma(xlow, r)

            denoised_low = self.noise(xlow, r)
            est_high = self.enhance(denoised_low, r)
            return est_high, r, denoised_low      

    def enhance(self, ilow, rx):
        ienhance_image = torch.pow(ilow+1e-6, rx)
        return ienhance_image
