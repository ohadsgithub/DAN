import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PCAEncoder
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DPCB_bottleneck(nn.Module):
    def __init__(self, nf, ksize, expand_ratio):
        super().__init__()
        hidden_dim = round(nf * expand_ratio)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(nf, hidden_dim, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.5, True), #unnecessary relu? 0.2 or 0.5?
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, ksize, 1, ksize // 2, groups=hidden_dim),
            nn.LeakyReLU(0.2, True),
            # pw-linear
            nn.Conv2d(hidden_dim, nf, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)

class DPCB_MNv2_simple_half(nn.Module):
    def __init__(self, nf, ksize=3, expand_ratio=1.): # , width_mult=1.
        super().__init__()
        #super(InvertedResidual, self).__init__()

        hidden_dim = round(nf * expand_ratio)

        if expand_ratio == 1:
            self.body = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, ksize, 1, ksize // 2, groups=hidden_dim),
                nn.LeakyReLU(0.5, True), #unnecessary relu? 0.2 or 0.5?
                # pw-linear
                nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),

                nn.LeakyReLU(0.2, True),

                nn.Conv2d(hidden_dim, hidden_dim, ksize, 1, ksize // 2, groups=hidden_dim),
                nn.LeakyReLU(0.5, True), #unnecessary relu? 0.2 or 0.5?
                # pw-linear
                nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            )
        else:
            self.body = nn.Sequential(
                DPCB_bottleneck(self, nf, ksize, expand_ratio),
                #put batch norm here?

                # pw
                nn.Conv2d(nf, hidden_dim, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.5, True), #unnecessary relu? 0.2 or 0.5?
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, ksize, 1, ksize // 2, groups=hidden_dim),
                nn.LeakyReLU(0.2, True),
                # pw-linear
                nn.Conv2d(hidden_dim, nf, 1, 1, 0, bias=False),
            )
    def forward(self, x):
        return self.body(x)

class DPCB_MNv2_simple(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=1, expand_ratio_B=1., expand_ratio_C=1.): # , width_mult=1.
        super().__init__()
        #super(InvertedResidual, self).__init__()

        #hidden_dim1 = round(nf1 * expand_ratio_B)
        #hidden_dim2 = round(nf2 * expand_ratio_C)
        #self.identity1 = expand_ratio_C != 1
        #self.identity2 = expand_ratio_B != 1

        self.body1 = DPCB_MNv2_simple_half(nf1, ksize1, expand_ratio_B) # need sequential?

        if ksize2 == 1:
            self.body2 = nn.Sequential(
                nn.Conv2d(nf2, nf1, ksize2, 1, ksize2 // 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf1, nf1, ksize2, 1, ksize2 // 2),
            )
        else:
            self.body2 = DPCB_MNv2_simple_half(nf2, ksize2, expand_ratio_C)  # need sequential?

    def forward(self, x):

        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        x[0] = x[0] + torch.mul(f1, f2)
        x[1] = x[1] + f2
        return x

class DPCG_MNv2_simple(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb, expand_ratio_B=1., expand_ratio_C=1.):
        super().__init__()

        self.body = nn.Sequential(*[DPCB_MNv2_simple(nf1, nf2, ksize1, ksize2, expand_ratio_B, expand_ratio_C) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


class STD(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        scale = self.scale
        B, C, H, W = x.shape
        x = x.view(B, C, H // scale, scale, W // scale, scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * scale ** 2, H // scale, W // scale)
        return x


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        B, C, H, W = x.shape
        size = self.size

        ch = H // 2
        cw = W // 2

        sth = ch - size // 2
        edh = ch + size // 2

        stw = cw - size // 2
        edw = cw + size // 2

        x = x[:, :, sth:edh, stw:edw]

        return xs # return x?


class Estimator(nn.Module):
    def __init__(
        self, in_nc=1, nf=64, para_len=10, num_blocks=5, scale=4, kernel_size=4, width_mult_E_B=1. , width_mult_E_C=1. , expand_ratio_E_B=1., expand_ratio_E_C=1.
    ):
        super(Estimator, self).__init__()

        #nf= = _make_divisible(nf * width_mult_E, 4 if width_mult == 0.1 else 8)
        nf_B= _make_divisible(nf * width_mult_E_B, 4 if width_mult_E_B == 0.1 else 8)
        nf_C= _make_divisible(nf * width_mult_E_C, 4 if width_mult_E_C == 0.1 else 8)

        self.ksize = kernel_size

        self.head_LR = nn.Sequential(
            # CenterCrop(self.ksize + scale),
            nn.Conv2d(in_nc, nf_B // 2, 5, 1, 2)
        )
        self.head_HR = nn.Sequential(
            # CenterCrop(self.ksize + scale),
            nn.Conv2d(in_nc, nf_C // 2, scale * 4 + 1, scale, scale * 2),
        )

        #num_blocks = 5

        #why // 2 ?
        self.body = DPCG_MNv2_simple(nf_B // 2, nf_C // 2, 3, 3, num_blocks, expand_ratio_E_B, expand_ratio_E_C)

        self.tail = nn.Sequential(
            nn.Conv2d(nf_B // 2, nf, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, self.ksize ** 2, 1, 1, 0),
            nn.Softmax(1),
        )

    def forward(self, GT, LR):
        lrf = self.head_LR(LR)
        hrf = self.head_HR(GT)

        f = [lrf, hrf]
        f, _ = self.body(f)
        f = self.tail(f)

        return f.view(*f.size()[:2])


class Restorer(nn.Module):
    def __init__(
        self, in_nc=1, nf=64, nb=8, ng=1, scale=4, input_para=10, min=0.0, max=1.0, width_mult_R_B=1. , width_mult_R_C=1., expand_ratio_R_B=1., expand_ratio_R_C=1.
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        #assert expand_ratio_R_C
        expand_ratio_R_C=1.

        # nf= = _make_divisible(nf * width_mult_E, 4 if width_mult == 0.1 else 8)
        nf_B = _make_divisible(nf * width_mult_R_B, 4 if width_mult_R_B == 0.1 else 8)
        nf_C = _make_divisible(nf * width_mult_R_C, 4 if width_mult_R_C == 0.1 else 8)

        out_nc = in_nc
        #in_nc=1????

        self.head1 = nn.Conv2d(in_nc, nf_B, 3, stride=1, padding=1) #LR - basic
        self.head2 = nn.Conv2d(input_para, nf_C, 1, 1, 0) #kernel - conditional

        #here its not //2
        body = [DPCG_MNv2_simple(nf_B, nf_C, 3, 1, nb, expand_ratio_R_B, expand_ratio_R_C) for _ in range(ng)]
        self.body = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf_B, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        elif scale == 1:
            self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)

        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )

    def forward(self, input, ker_code):
        B, C, H, W = input.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1))

        f1 = self.head1(input)
        f2 = self.head2(ker_code_exp)
        inputs = [f1, f2]
        f, _ = self.body(inputs)
        f = self.fusion(f)
        out = self.upscale(f)

        return out  # torch.clamp(out, min=self.min, max=self.max)


class DAN(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=16,
        ng=5,
        in_nc=3,
        upscale=4,
        input_para=10,
        kernel_size=21,
        loop=8,
        pca_matrix_path=None,
    ):
        super(DAN, self).__init__()

        #assert nb #####################################
        #assert ng #################
        #loop ###################
        #estimator_nb
        width_mult_R_B=1.
        width_mult_R_C=1.
        width_mult_E_B=1.
        width_mult_E_C=1.
        expand_ratio_R_B=1.
        expand_ratio_R_C=1. #should always be 1?
        expand_ratio_E_B=1.
        expand_ratio_E_C=1.


        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.Restorer = Restorer(
            nf=nf, in_nc=in_nc, nb=nb, ng=ng, scale=self.scale, input_para=input_para, width_mult_R_B=width_mult_R_B,
            width_mult_R_C=width_mult_R_C, expand_ratio_R_B=expand_ratio_R_B, expand_ratio_R_C=expand_ratio_R_C
        )
        self.Estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, scale=self.scale, width_mult_E_B=width_mult_E_B,
            width_mult_E_C=width_mult_E_C, expand_ratio_E_B=expand_ratio_E_B, expand_ratio_E_C=expand_ratio_E_C
        )

        self.register_buffer("encoder", torch.load(pca_matrix_path)[None])

        kernel = torch.zeros(1, self.ksize, self.ksize)
        kernel[:, self.ksize // 2, self.ksize // 2] = 1

        self.register_buffer("init_kernel", kernel)
        init_ker_map = self.init_kernel.view(1, 1, self.ksize ** 2).matmul(
            self.encoder
        )[:, 0]
        self.register_buffer("init_ker_map", init_ker_map)

    def forward(self, lr):

        srs = []
        ker_maps = []
        kernels = []

        B, C, H, W = lr.shape
        ker_map = self.init_ker_map.repeat([B, 1])

        for i in range(self.loop):

            sr = self.Restorer(lr, ker_map.detach())
            kernel = self.Estimator(sr.detach(), lr)
            ker_map = kernel.view(B, 1, self.ksize ** 2).matmul(self.encoder)[:, 0]

            srs.append(sr)
            ker_maps.append(ker_map)
            kernels.append(kernel)
        return [srs, ker_maps, kernels]
