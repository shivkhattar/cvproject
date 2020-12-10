import torch
import torch.nn.functional as F
from torchvision import models

from collections import OrderedDict
import torch.nn as nn


class _ActivatedBatchNorm(nn.Module):
    def __init__(self, num_features, activation='relu', slope=0.01, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=False)
        bn_depth = nn.BatchNorm2d(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)
        bn_point = nn.BatchNorm2d(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    ]))

    def forward(self, x):
        return self.block(x)


ActivatedBatchNorm = _ActivatedBatchNorm


class SegmentatorTTA(object):
    @staticmethod
    def hflip(x):
        return x.flip(3)

    @staticmethod
    def vflip(x):
        return x.flip(2)

    @staticmethod
    def trans(x):
        return x.transpose(2, 3)

    def pred_resize(self, x, size, net_type='unet'):
        h, w = size
        if net_type == 'unet':
            pred = self.forward(x)
            if x.shape[2:] == size:
                return pred
            else:
                return F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        else:
            pred = self.forward(F.pad(x, (0, 1, 0, 1)))
            return F.interpolate(pred, size=(h + 1, w + 1), mode='bilinear', align_corners=True)[..., :h, :w]

    def tta(self, x, scales=None, net_type='unet'):
        size = x.shape[2:]
        if scales is None:
            seg_sum = self.pred_resize(x, size, net_type)
            seg_sum += self.hflip(self.pred_resize(self.hflip(x), size, net_type))
            return seg_sum / 2
        else:
            # scale = 1
            seg_sum = self.pred_resize(x, size, net_type)
            seg_sum += self.hflip(self.pred_resize(self.hflip(x), size, net_type))
            for scale in scales:
                scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
                seg_sum += self.pred_resize(scaled, size, net_type)
                seg_sum += self.hflip(self.pred_resize(self.hflip(scaled), size, net_type))
            return seg_sum / ((len(scales) + 1) * 2)


def create_encoder(enc_type, pretrained=True):
    return resnet(enc_type, pretrained)


def resnet(name, pretrained=False):
    def get_channels(layer):
        block = layer[-1]
        if isinstance(block, models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, models.resnet.Bottleneck):
            return block.conv3.out_channels
        raise RuntimeError("unknown resnet block: {}".format(block))

    resnet = models.resnet50(pretrained=pretrained)

    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    layer0.out_channels = resnet.bn1.num_features
    resnet.layer1.out_channels = get_channels(resnet.layer1)
    resnet.layer2.out_channels = get_channels(resnet.layer2)
    resnet.layer3.out_channels = get_channels(resnet.layer3)
    resnet.layer4.out_channels = get_channels(resnet.layer4)
    return [layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]


def create_decoder():
    return DecoderUnetSCSE


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)

        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class DecoderUnetSCSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(middle_channels),
            SCSEBlock(middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, *args):
        x = torch.cat(args, 2)
        return self.block(x)


class UNet(nn.Module, SegmentatorTTA):
    def __init__(self, output_channels=21, enc_type='resnet50', dec_type='unet_scse',
                 num_filters=16, pretrained=False):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type

        assert enc_type in ['resnet50']
        assert dec_type in ['unet_scse']

        encoder = create_encoder(enc_type, pretrained)
        Decoder = create_decoder()

        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]

        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 32 * 2, num_filters * 32)

        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 32, num_filters * 32 * 2,
                                num_filters * 16)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 16, num_filters * 16 * 2,
                                num_filters * 8)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        self.decoder1 = Decoder(self.encoder1.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)

        self.logits = nn.Sequential(
            nn.Conv2d(num_filters * (16 + 8 + 4 + 2 + 1), 64, kernel_size=1, padding=0),
            ActivatedBatchNorm(64),
            nn.Conv2d(64, self.output_channels, kernel_size=1)
        )

    def forward(self, x):
        img_size = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        c = self.center(self.pool(e5))
        e1_up = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False)

        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1_up)

        u5 = F.interpolate(d5, img_size, mode='bilinear', align_corners=False)
        u4 = F.interpolate(d4, img_size, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, img_size, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, img_size, mode='bilinear', align_corners=False)

        # Hyper column
        d = torch.cat((d1, u2, u3, u4, u5), 1)
        logits = self.logits(d)

        return logits
