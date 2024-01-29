""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from .unet_parts import *
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=7, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    # def get_high(self,im_tensor):
    #     fft_src_np = torch.fft.fftn(im_tensor, dim=(-4, -3, -2, -1))
    #     fshift = torch.fft.fftshift(fft_src_np).cuda()
    #     np_zero = torch.ones_like(fshift).cuda()
    #     b = im_tensor.shape[2]//2
    #     c = im_tensor.shape[3]//2
    #     s_b = b//20
    #     s_c = c//20
    #     if s_b == 0:
    #         s_b = 1
    #         s_c = 1
    #     np_zero[:, :, b - s_b:b + s_b, c - s_c:c + s_c] = 0
    #     fshift = fshift * np_zero
    #     ishift = torch.fft.ifftshift(fshift).cuda()
    #     iimg = abs(torch.fft.ifftn(ishift, dim=(-4, -3, -2, -1)).cuda())
    #     return iimg
    def get_high(self,im_tensor):
        fft_src_np = torch.fft.fftn(im_tensor, dim=(-4, -3, -2, -1))
        fshift = torch.fft.fftshift(fft_src_np)
        np_zero = torch.ones_like(fshift)
        b = im_tensor.shape[2]//2
        c = im_tensor.shape[3]//2
        s_b = b//20
        s_c = c//20
        if s_b == 0:
            s_b = 1
            s_c = 1
        np_zero[:, :, b - s_b:b + s_b, c - s_c:c + s_c] = 0
        fshift = fshift * np_zero
        ishift = torch.fft.ifftshift(fshift)
        iimg = abs(torch.fft.ifftn(ishift, dim=(-4, -3, -2, -1)))
        return iimg

    def forward(self, x):
        x1 = self.inc(x)
        # x1=self.get_high(x1)
        x2 = self.down1(x1)
        # x2 = self.get_high(x2)
        x3 = self.down2(x2)
        # x3 = self.get_high(x3)
        x4 = self.down3(x3)
        # x4 = self.get_high(x4)
        x5 = self.down4(x4)
        # x5 = self.get_high(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits,x5

