import torch
import torch.nn as nn
from .resnet import resnet18


class PvnetModelResnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(PvnetModelResnet18, self).__init__()
        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            # SimamAttention(),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            # SimamAttention(),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            # SimamAttention(),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            # SimamAttention(),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            # SimamAttention(),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
    def forward(self, x):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        if fm.shape[2] == 136:
            fm = nn.functional.interpolate(fm, (135, 180), mode='bilinear', align_corners=False)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        x = self.convraw(torch.cat([fm, x], 1))
        seg_pred = x[:, :self.seg_dim, :, :]
        ver_pred = x[:, self.seg_dim:, :, :]

        # seg_pred = feature_alignment['mask'].cuda()
        # seg_pred = torch.cat([1-seg_pred, seg_pred], dim=0).unsqueeze(0)
        ret = {'seg': seg_pred, 'vertex': ver_pred}
        return ret