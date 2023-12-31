import torch
import torch.nn as nn
import numpy as np
from model.backbone_ex import Backbone
from model.neck import Neck
from model.head import Head
from model.Flayer import FLayer



class shipmodel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        output_ch = (5 + 1 + n_classes) * 3 * 6
        radian = np.pi / 180
        angles = [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90]
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(output_ch)
        self.Flayer1 = FLayer(num_classes=n_classes, anchors=[[12, 16], [19, 36], [40, 28]],
                               angles=angles, stride=8, scale_x_y=1.2, ignore_thresh=0.6)
        self.Flayer2 = FLayer(num_classes=n_classes, anchors=[[36, 75], [76, 55], [72, 146]],
                               angles=angles, stride=16, scale_x_y=1.1, ignore_thresh=0.6)
        self.Flayer3 = FLayer(num_classes=n_classes, anchors=[[142, 110], [192, 243], [459, 401]],
                               angles=angles, stride=32, scale_x_y=1.05, ignore_thresh=0.6)

    def forward(self, i, target=None):
        if target is None:
            inference = True
        else:
            inference = False

        d2, d3, d4, d5 = self.backbone(i)
        x20, x13, x6 = self.neck(d2, d5, d4, d3, inference)
        x2, x10, x18 = self.head(x20, x13, x6)
        y1, loss1 = self.Flayer1(x2, target)
        y2, loss2 = self.Flayer2(x10, target)
        y3, loss3 = self.Flayer3(x18, target)

        return torch.cat([y1, y2, y3], 1), (loss1 + loss2 + loss3)

model = shipmodel()
