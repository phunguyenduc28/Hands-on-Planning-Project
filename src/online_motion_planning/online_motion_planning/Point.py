import math
import numpy as np


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def dist(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    def numpy(self):
        return np.array([self.x, self.y])

    def dist_line(self, l):
        return np.linalg.norm(
            np.cross(l.p2.numpy() - l.p1.numpy(), l.p1.numpy() - self.numpy())
        ) / np.linalg.norm(l.p2.numpy() - l.p1.numpy())

    def __str__(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))

    def to_str(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))

    def dot(self, p):
        return self.x * p.x + self.y * p.y

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)

    def vector(self, p):
        return Point(p.x - self.x, p.y - self.y)

    def length(self):
        # lenght of vector
        return math.sqrt((self.x) ** 2 + (self.y) ** 2)
    def unit(self):
        mag = self.length()
        if mag > 0:
            return Point(self.x / mag, self.y / mag)
        else:
            return Point(0, 0)

    def scale(self, sc):
        return Point(self.x * sc, self.y * sc)

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __truediv__(self, s):
        return Point(self.x / s, self.y / s)

    def __floordiv__(self, s):
        return Point(int(self.x / s), int(self.y / s))

    def __mul__(self, s):
        return Point(self.x * s, self.y * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def dist_segment(self, s):
        line_vec = s.p1.vector(s.p2)
        pnt_vec = s.p1.vector(self)
        line_len = line_vec.length()
        line_unitvec = line_vec.unit()
        pnt_vec_scaled = pnt_vec.scale(1.0 / line_len)
        t = line_unitvec.dot(pnt_vec_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = line_vec.scale(t)
        dist = nearest.dist(pnt_vec)
        nearest = nearest.add(s.p1)
        return dist
