import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import random


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]),
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False):
        super(Discriminator, self).__init__()
        self.nc = nc
        assert(imsize==32 or imsize==64 or imsize==128)
        self.im_size = imsize
        self.hflip = hflip

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False), 
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])

        self.rf_big = nn.Sequential(
                            conv2d(nfc[128] , nfc[32], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[32]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[32], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[256])
        self.se_4_32 = SEBlock(nfc[256], nfc[128])
        
        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[512], 3, 1, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[512],  nfc[256]))
                                            
        self.rf_small = conv2d(nfc[256], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[128], nc)
        self.decoder_part = SimpleDecoder(nfc[256], nc)
        self.decoder_small = SimpleDecoder(nfc[256], nc)

    def forward(self, input, y=None):
        input = input[:, :self.nc]
        input = input.view(-1, self.im_size, self.im_size, self.nc).permute(0, 3, 1, 2)

        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input),1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        if type(input) is not list:
            input = [F.interpolate(input, size=self.im_size),
                     F.interpolate(input, size=(input.size(-1)//2))]

        feat_2 = self.down_from_big(input[0])    
        feat_4 = self.down_4(feat_2)
        feat_16 = self.se_2_16(feat_2, feat_4)
        
        feat_8 = self.down_8(feat_16)
        feat_last = self.se_4_32(feat_4, feat_8)

        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(input[1])
        rf_1 = self.rf_small(feat_small).view(-1)

        if y[0] == 1:    
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            part = random.randint(0, 3)
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_16[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_16[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_16[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_16[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]).mean(), [rec_img_big, rec_img_small, rec_img_part], part, input[0]

        return torch.cat([rf_0, rf_1]).mean()
