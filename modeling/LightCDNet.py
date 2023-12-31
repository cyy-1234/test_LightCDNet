import torch
import torch.nn as nn

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


def basic_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

    )


class LightCDNet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(LightCDNet, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn
        if backbone == 'resnet':
            self.lowf_conv = basic_block(256 * 2, 256 * 1)
            self.highf_conv = basic_block(2048 * 2, 2048 * 1)

        if backbone == 'xception':
            self.lowf_conv = basic_block(128 * 2, 128 * 1)
            self.highf_conv = basic_block(2048 * 2, 2048 * 1)
        if backbone == 'mobilenet':
            self.lowf_conv = basic_block(24 * 2, 24 * 1)
            self.highf_conv = basic_block(320 * 2, 320 * 1)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.block1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(128),
                                    nn.ReLU(),

                                    )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(64),
                                    nn.ReLU(),
                                    )
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, input1, input2):
        # -------------------SiameseEncoder------------------------
        # shaing weight
        FH1, FL1 = self.backbone(input1)
        FH2, FL2 = self.backbone(input2)
        # -------------------MultitemporalFeatureFusion-------------------------
        # low level features fusion
        FLC = torch.cat((FL1, FL2), dim=1)
        FLC1 = self.lowf_conv(FLC)
        # high level features fusion
        FHC = torch.cat((FH1, FH2), dim=1)
        FHC1 = self.highf_conv(FHC)

        FHASPP = self.aspp(FHC1)

        # The first step of the decoder is in self.decoder(modeling/decoder.py)

        FD1 = self.decoder(FHASPP, FLC1)

        # Deconvolution
        up1 = self.up1(FD1)
        block1 = self.block1(up1)
        up2 = self.up2(block1)
        block2 = self.block2(up2)

        F = self.conv(block2)  # classify
        return F

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder, self.lowf_conv, self.highf_conv, self.up1, self.block1, self.up2,
                   self.block2, self.conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = LightCDNet(backbone='mobilenet', output_stride=16)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))



