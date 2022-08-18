import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from decoder import DecoderBlock, Upsample
from module import CPGmodule, CoarseSeg


class CPGnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CPGnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])

        self.seg0 = CoarseSeg(filters[0], n_classes)
        self.seg1 = CoarseSeg(filters[0], n_classes)
        self.seg2 = CoarseSeg(filters[1], n_classes)
        self.seg3 = CoarseSeg(filters[2], n_classes)
        self.seg4 = CoarseSeg(filters[3], n_classes)

        self.cpg1 = CPGmodule(filters[0], filters[0])
        self.cpg2 = CPGmodule(filters[1], filters[1])
        self.cpg3 = CPGmodule(filters[2], filters[2])
        self.cpg4 = CPGmodule(filters[3], filters[3])

        self.coarse_cat = nn.Sequential(
            nn.Conv2d(3, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.fine_cat = nn.Sequential(
            nn.Conv2d(filters[0] + filters[1] + filters[2], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.Conv2d(filters[0], n_classes, kernel_size=1)
        )

        self.upx2 = Upsample(scale_factor=2)
        self.upx4 = Upsample(scale_factor=4)
        self.upx8 = Upsample(scale_factor=8)
        self.upx16 = Upsample(scale_factor=16)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Generate coarse segmentation
        coarse3 = self.seg3(e3)
        coarse2 = self.seg2(e2)
        coarse1 = self.seg1(e1)

        # CPG module
        cpg3 = self.cpg3(coarse3, e3, e4)
        cpg2 = self.cpg2(coarse2, e2, e3)
        cpg1 = self.cpg1(coarse1, e1, e2)

        # Decoder
        d3 = self.decoder3(e4) + cpg3
        d2 = self.decoder2(d3) + cpg2
        d1 = self.decoder1(d2) + cpg1

        coarse2 = self.upx2(coarse2)
        coarse3 = self.upx4(coarse3)

        d2 = self.upx2(d2)
        d3 = self.upx4(d3)
        fine_out = self.fine_cat(torch.cat((d3, d2, d1), 1))

        fine_out = self.upx4(fine_out)
        fine_out = F.sigmoid(fine_out)
        coarse1 = F.sigmoid(self.upx4(coarse1))
        coarse2 = F.sigmoid(self.upx4(coarse2))
        coarse3 = F.sigmoid(self.upx4(coarse3))

        return [fine_out, coarse1, coarse2, coarse3]