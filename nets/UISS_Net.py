import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.Auxiliary_net import Auxiliary_net

from torchsummary import summary


class SpatialAttention(nn.Module):
    def __init__(self, in_channl):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

        self.conv1 = nn.Conv2d(in_channl, in_channl * 2, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(in_channl * 2)
        self.bn2 = nn.BatchNorm2d(in_channl * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn(x0)
        avg_pool = torch.mean(x0, dim=1, keepdim=True)
        max_pool, _ = torch.max(x0, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = x0 * y.expand_as(x0)
        return y


from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UISS_NetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UISS_NetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class UISS_Net(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='resnet50'):
        super(UISS_Net, self).__init__()
        if backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            self.vgg = Auxiliary_net(pretrained=False)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use  resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = UISS_NetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = UISS_NetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = UISS_NetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = UISS_NetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.c1 = nn.Conv2d(2048, 1024, 3, 1, 1)
        self.b1 = nn.BatchNorm2d(1024)
        self.c2 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.b2 = nn.BatchNorm2d(512)
        self.c3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.b3 = nn.BatchNorm2d(256)
        self.c4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.b4 = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(512, 1024, 1)
        self.conv1 = nn.Conv2d(128, 64, 1)
        self.backbone = backbone
        self.a = 0.5
        self.con_down = nn.Conv2d(64, 256, 3, 2, 1)
        self.con_down11 = nn.Conv2d(512, 512, 1, 1, )
        self.con_down22 = nn.Conv2d(1024, 1024, 1, 1, )
        self.con_down33 = nn.Conv2d(2048, 2048, 1, 1, )
        self.con_down2 = nn.Conv2d(512, 256, 1, 1)

        self.con_down3 = nn.Conv2d(1024, 512, 1, 1)

        self.con_down4 = nn.Conv2d(2048, 1024, 1, 1)

        self.con_down5 = nn.Conv2d(4096, 2048, 1, 1)
        self.con_N1 = nn.Sequential(nn.Conv2d(64, 64, 1, 1), nn.ReLU(inplace=True))
        self.con_N2 = nn.Sequential(nn.Conv2d(128, 128, 1, 1), nn.ReLU(inplace=True))
        self.con_N3 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.ReLU(inplace=True))
        self.con_N4 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.ReLU(inplace=True))
        self.con_NC1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.con_NC2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.con_NC3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True))
        self.con_NC4 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1), nn.ReLU(inplace=True))
        self.se1 = SELayer(64)
        self.se2 = SELayer(256)
        self.se3 = SELayer(512)
        self.se4 = SELayer(1024)
        self.se5 = SELayer(2048)
        self.D1 = SpatialAttention(256)
        self.D2 = SpatialAttention(512)
        self.D3 = SpatialAttention(1024)
        self.relu = nn.ReLU(inplace=True)
        self.c_d1 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.ReLU(inplace=True))
        self.c_d2 = nn.Sequential(nn.Conv2d(512, 256, 1, 1), nn.ReLU(inplace=True))
        self.c_d3 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.ReLU(inplace=True))
        self.cc_d3 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.ReLU(inplace=True))

    def forward(self, inputs):
        if self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
            [feat11, feat22, feat33, feat44] = self.vgg.forward(inputs)
        feat1 = self.se1(feat1)
        feat2 = self.se2(feat2)
        feat3 = self.se3(feat3)
        feat4 = self.se4(feat4)
        feat5 = self.se5(feat5)


        feat4 = torch.cat([feat4, feat44], 1)
        feat4 = self.b1(self.c1(feat4))

        feat3 = torch.cat([feat3, feat33], 1)
        feat3 = self.b2(self.c2(feat3))
        feat2 = torch.cat([feat2, feat22], 1)
        feat2 = self.b3(self.c3(feat2))
        feat1 = torch.cat([feat1, feat11], 1)
        feat1 = self.b4(self.c4(feat1))

        up4 = self.up_concat4(feat4, feat5)

        x1 = nn.functional.interpolate(up4, size=feat3.size()[2:], mode='bilinear', align_corners=False)
        feat3 = torch.cat([x1, feat3], 1)
        feat3 = self.c_d1(feat3)

        up3 = self.up_concat3(feat3, up4)

        x2 = nn.functional.interpolate(up3, size=feat2.size()[2:], mode='bilinear', align_corners=False)
        feat2 = torch.cat([x2, feat2], 1)
        feat2 = self.c_d2(feat2)

        up2 = self.up_concat2(feat2, up3)

        x3 = nn.functional.interpolate(up2, size=feat1.size()[2:], mode='bilinear', align_corners=False)

        x3 = self.cc_d3(x3)

        feat1 = torch.cat([x3, feat1], 1)

        feat1 = self.c_d3(feat1)

        up1 = self.up_concat1(feat1, up2)
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


if __name__ == '__main__':
    net = UISS_Net(pretrained=False)
    # net=SpatialAttention(in_channl=3)
    net.cuda()
    summary(net, (3, 512, 512))