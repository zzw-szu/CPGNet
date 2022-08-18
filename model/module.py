import torch
import torch.nn as nn


class CPGmodule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CPGmodule, self).__init__()
        self.mid_channel = in_channels // 4
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, self.mid_channel, 1, bias=False),
                                   nn.BatchNorm2d(self.mid_channel),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(self.mid_channel, out_channels, kernel_size=1, stride=1, padding=0)
        self.upconv = nn.ConvTranspose2d(in_channels * 2, self.mid_channel, 3, stride=2, padding=1, output_padding=1)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.init_weight()

    def forward(self, coarse_seg, ll_feature, hl_feature):
        b_coa, n_coa, h_coa, w_coa = coarse_seg.size()

        # context prior representation
        ll_feature = self.conv1(ll_feature)
        b_fea, c_fea, h_fea, w_fea = ll_feature.size()
        coarse_seg = coarse_seg.view(b_coa, n_coa, -1)
        ll_feature_rs = ll_feature.view(b_fea, c_fea, -1).permute(0, 2, 1)
        context = torch.bmm(coarse_seg, ll_feature_rs)
        F_context = torch.max(context, -1, keepdim=True)[0].expand_as(context) - context

        # semantic complement flow
        F_scf = self.GAP(self.upconv(hl_feature)).squeeze(-1).permute(0, 2, 1)
        attention = self.softmax(F_scf * F_context) + self.softmax(F_context)
        attention = attention.permute(0, 2, 1)

        guidance = torch.bmm(attention, coarse_seg)
        guidance = guidance.view(b_coa, c_fea, h_coa, w_coa)
        rf_feature = guidance + ll_feature
        rf_feature = self.conv2(rf_feature)

        return rf_feature

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class CoarseSeg(nn.Module):
    def __init__(self, ch_in, num_class):
        super(CoarseSeg, self).__init__()
        self.seg = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, 1, bias=False),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(False),
            nn.Dropout2d(0.5, False),
            nn.Conv2d(ch_in // 2, ch_in // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(False),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(ch_in // 2, num_class, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return self.seg(x)
