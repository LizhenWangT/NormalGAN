import torch
from torch import nn
import collections
import torchvision
import numpy as np
import ops


class PatchDisNet(nn.Module):
    def __init__(self, channel, ngf):
        super(PatchDisNet, self).__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(nn.Conv2d(channel, ngf, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True))

        self.conv2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(ngf * 2))

        self.conv3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(ngf * 4))

        self.conv4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(ngf * 8))

        self.conv5 = nn.Sequential(nn.Conv2d(ngf * 8, 1, kernel_size=kw, stride=1, padding=padw))


        for m in self.modules():
            ops.weights_init(m)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return [out1, out2, out3, out4, out5]


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, ngf, upconv=False, norm=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        self.norm = norm
        self.upconv = upconv

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
        if self.upconv:
            self.u3 = nn.ConvTranspose2d(self.ngf * 16, self.ngf * 16, 3, padding=1, output_padding=1, stride=2)
            self.u2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 3, padding=1, output_padding=1, stride=2)
            self.u1 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, padding=1, output_padding=1, stride=2)
            self.u0 = nn.ConvTranspose2d(self.ngf * 2, self.ngf * 2, 3, padding=1, output_padding=1, stride=2)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data, inter_mode='nearest'):
        x0 = self.l0(input_data)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)
        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        if self.upconv:
            x3u = nn.functional.interpolate(self.u3(x3), size=x2.shape[2:4], mode=inter_mode)
        else:
            x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        if self.norm:
            x3u = self.n3u(x3u)

        if self.upconv:
            x2u = nn.functional.interpolate(self.u2(x3u), size=x1.shape[2:4], mode=inter_mode)
        else:
            x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)

        if self.upconv:
            x1u = nn.functional.interpolate(self.u1(x2u), size=x0.shape[2:4], mode=inter_mode)
        else:
            x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        if self.norm:
            x1u = self.n1u(x1u)

        if self.upconv:
            x0u = nn.functional.interpolate(self.u0(x1u), size=input_data.shape[2:4], mode=inter_mode)
        else:
            x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u


class SUNet(nn.Module):
    def __init__(self, in_channel, out_channel, ngf, upconv=False, norm=False):
        super(SUNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        self.norm = norm
        self.upconv = upconv

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
        if self.upconv:
            self.u3 = nn.ConvTranspose2d(self.ngf * 16, self.ngf * 16, 3, padding=1, output_padding=1, stride=2)
            self.u2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 3, padding=1, output_padding=1, stride=2)
            self.u1 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, padding=1, output_padding=1, stride=2)
            self.u0 = nn.ConvTranspose2d(self.ngf * 2, self.ngf * 2, 3, padding=1, output_padding=1, stride=2)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data, inter_mode='nearest'):
        x0 = self.l0(input_data)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)
        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        if self.upconv:
            x3u = nn.functional.interpolate(self.u3(x3), size=x2.shape[2:4], mode=inter_mode)
        else:
            x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        if self.norm:
            x3u = self.n3u(x3u)

        if self.upconv:
            x2u = nn.functional.interpolate(self.u2(x3u), size=x1.shape[2:4], mode=inter_mode)
        else:
            x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)

        if self.upconv:
            x1u = nn.functional.interpolate(self.u1(x2u), size=x0.shape[2:4], mode=inter_mode)
        else:
            x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(x1u)
        if self.norm:
            x1u = self.n1u(x1u)

        if self.upconv:
            x0u = nn.functional.interpolate(self.u0(x1u), size=input_data.shape[2:4], mode=inter_mode)
        else:
            x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u


class FUNet(nn.Module):
    def __init__(self, in_channel, out_channel, ngf, upconv=False, norm=False):
        super(FUNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        self.norm = norm
        self.upconv = upconv

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.nn0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.nn1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.nn2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.nn3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n0u = torch.nn.InstanceNorm2d(self.ngf)
        if self.upconv:
            self.u3 = nn.ConvTranspose2d(self.ngf * 32, self.ngf * 32, 3, padding=1, output_padding=1, stride=2)
            self.u2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 3, padding=1, output_padding=1, stride=2)
            self.u1 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, padding=1, output_padding=1, stride=2)
            self.u0 = nn.ConvTranspose2d(self.ngf * 2, self.ngf * 2, 3, padding=1, output_padding=1, stride=2)
            self.ux = nn.ConvTranspose2d(self.ngf * 1, self.ngf * 1, 3, padding=1, output_padding=1, stride=2)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size -> size / 2
        self.ll0 = nn.Sequential(
            nn.Conv2d(9, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.ll1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 4 -> size / 8
        self.ll2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.ll3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 32, self.ngf * 32, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 32, self.ngf * 32, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 32, self.ngf * 32, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 32, self.ngf * 32, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 40, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            # nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            # nn.ELU()
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        # size -> size * 2
        self.lxu = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf // 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf // 2, self.ngf // 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf // 2, self.out_channel, 1, padding=0, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data, light_data, inter_mode='nearest'):
        H, W = input_data.shape[2:4]
        x0 = self.l0(input_data)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)
        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)
        d0 = self.ll0(light_data)
        if self.norm:
            d0 = self.nn0(d0)
        d1 = self.ll1(d0)
        if self.norm:
            d1 = self.nn1(d1)
        d2 = self.ll2(d1)
        if self.norm:
            d2 = self.nn2(d2)
        d3 = self.ll3(d2)
        if self.norm:
            d3 = self.nn3(d3)
        x3 = torch.cat((x3, d3), dim=1)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        if self.upconv:
            x3u = nn.functional.interpolate(self.u3(x3), size=x2.shape[2:4], mode=inter_mode)
        else:
            x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        if self.norm:
            x3u = self.n3u(x3u)

        if self.upconv:
            x2u = nn.functional.interpolate(self.u2(x3u), size=x1.shape[2:4], mode=inter_mode)
        else:
            x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)

        if self.upconv:
            x1u = nn.functional.interpolate(self.u1(x2u), size=x0.shape[2:4], mode=inter_mode)
        else:
            x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        if self.norm:
            x1u = self.n1u(x1u)

        if self.upconv:
            x0u = nn.functional.interpolate(self.u0(x1u), size=input_data.shape[2:4], mode=inter_mode)
        else:
            x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        # if self.norm:
        #     x0u = self.n0u(x0u)

        # if self.upconv:
        #     xu = nn.functional.interpolate(self.ux(x0u), size=[H * 2, W * 2], mode=inter_mode)
        # else:
        #     xu = nn.functional.interpolate(x0u, size=[H * 2, W * 2], mode=inter_mode)
        # xu = self.lxu(xu)
        return x0u

