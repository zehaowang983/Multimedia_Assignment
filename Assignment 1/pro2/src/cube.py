from operator import itemgetter
import numpy as np

class ColorSpace(object):
    rmax = 255.
    rmin = 0.
    gmax = 255.
    gmin = 0.
    bmax = 255.
    bmin = 0.

    def __init__(self, *colors):
        self._colors = colors or []
        self.color_size()

    @property
    def colors(self):
        return self._colors

    # 每个颜色通道的极差
    @property
    def size(self):
        return self.rmax - self.rmin, self.gmax - self.gmin, self.bmax - self.bmin

    # 每个cube的RGB平均值
    @property
    def average(self):
        # 所有像素点的数量
        length = len(self.colors)
        # 提取每个通道的值
        r = [c[0] for c in self.colors]
        g = [c[1] for c in self.colors]
        b = [c[2] for c in self.colors]
        # 取每个通道的平均值
        r = sum(r) / length
        g = sum(g) / length
        b = sum(b) / length
        # print(r,g,b)
        r = round(r)
        g = round(g)
        b = round(b)
        return r, g, b

    def color_size(self):
        R = [c[0] for c in self.colors]
        G = [c[1] for c in self.colors]
        B = [c[2] for c in self.colors]
        self.rmin = min(R)
        self.rmax = max(R)
        self.gmin = min(G)
        self.gmax = max(G)
        self.bmin = min(B)
        self.bmax = max(B)

    # 按中值一分为二，其中axis代表要排序的颜色，0为r，1为g，2为b
    def split(self, axis):
        self.color_size()
        self._colors = sorted(self.colors, key=itemgetter(axis))

        # Find median
        med_idx = round(len(self.colors) / 2)

        # Create splits
        return (
            ColorSpace(*self.colors[:med_idx]),
            ColorSpace(*self.colors[med_idx:]
        ))