import torch, torch.nn as nn
class SELite(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(8, channels // r)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels, 1, bias=True),
                                nn.Sigmoid())
    def forward(self, x):
        w = self.fc(self.pool(x)); return x * w
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(); p = k//2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))
class C2f_SELite(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__(); c_ = c2//2
        self.cv1 = ConvBNAct(c1, c_, 1, 1); self.cv2 = ConvBNAct(c1, c_, 1, 1)
        self.m = nn.ModuleList([ConvBNAct(c_, c_, 3, 1) for _ in range(n)])
        self.fuse = ConvBNAct(c_*2, c2, 1, 1); self.attn = SELite(c2)
        self.shortcut = shortcut and (c1 == c2)
    def forward(self, x):
        y1 = self.cv1(x); y2 = self.cv2(x)
        for m in self.m: y2 = m(y2)
        y = torch.cat([y1,y2],1); y = self.fuse(y)
        if self.shortcut: y = y + x
        return self.attn(y)
