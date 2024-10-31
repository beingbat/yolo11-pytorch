import torch
from torch import nn

from models.conv import Conv
from models.block import Concat, C3k2, SPPF, C2PSA
from models.head import Detect
# from ultralytics.nn.modules import (
#     C2PSA,
#     SPPF,
#     C3k2,
#     Concat,
#     Conv,
#     Detect,
# )

class yolov11(nn.Module):
    # no of classes
    nc = 80
    max_det = 300

    def __init__(self, model_size='nano'):
        super(yolov11, self).__init__()

        nano = (1, 1, 0, 3)
        small = (2, 1, 0, 3)
        medium = (2, 2, 0, 2)
        large = (2, 2, 1, 2)
        xlarge = (3, 2, 1, 2)

        if model_size in locals():
            model_klm = locals()[model_size]
        else:
            raise ValueError("The model size provided is not valid option")

        # k 1 for n, | 2 for s, m, l, | 4 for x
        # l 1 for n, s, | 2 for m, l, x
        # m 0 for n, s, m, | 1 for l, x
        # n 2 for m, 3 for n, s, l, x
        k, l, m, n = model_klm

        # backbone
        self.l0 = Conv(3, 16*k*l, 3, 2)
        self.l1 = Conv(16*k*l, 32*k*l, 3, 2)
        self.l2 = C3k2(32*k*l, 64*k*l, 1+m, l>1, 0.25)
        self.l3 = Conv(64*k*l, 64*k*l, 3, 2)
        self.l4 = C3k2(64*k*l, 128*k*l, 1+m, l>1, 0.25)
        self.l5 = Conv(128*k*l, 128*k*l, 3, 2)
        self.l6 = C3k2(128*k*l, 128*k*l, 1+m, True)
        self.l7 = Conv(128*k*l, 256*k, 3, 2)
        self.l8 = C3k2(256*k, 256*k, 1+m, True)
        self.l9 = SPPF(256*k, 256*k, 5)
        self.l10 = C2PSA(256*k, 256*k, 1+m)
        # YOLO11n Head
        self.l11 = nn.Upsample(None, 2, 'nearest')
        self.l12 = Concat(1) # l11, l6
        self.l13 = C3k2(256*k+128*k*l, 128*k*l, 1+m, l>1)
        self.l14 = nn.Upsample(None, 2, 'nearest')
        self.l15 = Concat(1) # l14, l4
        self.l16 = C3k2(128*k*2*l, 64*k*l, 1+m, l>1)
        self.l17 = Conv(64*k*l, 64*k*l, 3, 2)
        self.l18 = Concat(1) # l17, l13
        self.l19 = C3k2(192*k*l, 128*k*l, 1+m, l>1)
        self.l20 = Conv(128*k*l, 128*k*l, 3, 2)
        self.l21 = Concat(1) # l20, l10
        self.l22 = C3k2(128*n*k*l, 256*k, 1+m, True)
        self.l23 = Detect(self.nc, [64*k*l, 128*k*l, 256*k]) # l16, l19, l22

    def forward(self, x):
        # backbone
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x4 = self.l4(x)
        x = self.l5(x4)
        x6 = self.l6(x)
        x = self.l7(x6)
        x = self.l8(x)
        x = self.l9(x)
        x10 = self.l10(x)
        # head
        x = self.l11(x10)
        x = self.l12([x, x6])
        x13 = self.l13(x)
        x = self.l14(x13)
        x = self.l15([x, x4])
        x16 = self.l16(x)
        x = self.l17(x16)
        x = self.l18([x, x13])
        x19 = self.l19(x)
        x = self.l20(x19)
        x = self.l21([x, x10])
        x = self.l22(x)
        x = self.l23([x16, x19, x])
        return x
