import math
from multiprocessing import Value


from multiprocessing.sharedctypes import Value
from ctypes import Structure, c_double, c_int


class StatStreamStruct(Structure):
    _fields_ = [
        ('sum', c_double),
        ('sum_sqr', c_double),
        ('first_obs', c_double),
        ('min', c_double),
        ('max', c_double),
        ('current_count', c_int),
        ('current_obs', c_double),
        ('drop_obs', c_int)
    ]


class StatStream(object):
    """
        Store the sum of the observations of the sum of the observation squared
        The first few observations are discarded (usually slower than the rest)

        The average and the standard deviation is computed at the user's request

        In order to make the computation stable we store the first observation and subtract it to every other
        observations. The idea is if x ~ N(mu, sigma)  x - x0 and the sum of x - x0 should be close(r) to 0 allowing
        for greater precision; without that trick `var` was getting negative on some iteration.
    """
    def __init__(self, drop_first_obs=10):
        self.struct = Value(StatStreamStruct,
            0,                      # sum
            0,                      # sum_sqr
            0,                      # first_obs
            float('+inf'),          # min
            float('-inf'),          # max
            0,                      # current_count
            0,                      # current_obs
            drop_first_obs)         # drop_obs

    @property
    def sum(self):
        return self.struct.sum

    @property
    def sum_sqr(self):
        return self.struct.sum_sqr

    @property
    def current_count(self):
        return self.struct.current_count

    @property
    def current_obs(self):
        return self.struct.current_obs

    @property
    def max(self):
        return self.struct.max

    @property
    def min(self):
        return self.struct.min

    @property
    def drop_obs(self):
        return self.struct.drop_obs

    @property
    def first_obs(self):
        return self.struct.first_obs

    def __iadd__(self, other):
        self.update(other, 1)
        return self

    def update(self, val, weight=1):
        self.struct.current_count += weight

        if self.current_count < self.drop_obs:
            self.struct.current_obs = val
            return

        if self.count == 1:
            self.struct.first_obs = val

        self.struct.current_obs = val - self.first_obs
        self.struct.sum += float(self.current_obs) * float(weight)
        self.struct.sum_sqr += float(self.current_obs * self.current_obs) * float(weight)

        self.struct.min = min(self.min, val)
        self.struct.max = max(self.max, val)

    def to_array(self, transform=None):
        if transform is not None:
            return [transform(self.avg), 'NA', transform(self.min), transform(self.max), self.count]
        return [self.avg, self.sd, self.min, self.max, self.count]

    @property
    def val(self) -> float:
        return self.current_obs + self.first_obs

    @property
    def count(self)-> int:
        # is count is 0 then self.sum is 0 so everything should workout
        return max(self.current_count - self.drop_obs, 1)

    @property
    def avg(self) -> float:
        return self.sum / float(self.count) + self.first_obs

    @property
    def var(self) -> float:
        avg = self.sum / float(self.count)
        return self.sum_sqr / float(self.count) - avg * avg

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)


if __name__ == '__main__':
    print(Value(StatStreamStruct,
        0,  # sum
        0,  # sum_sqr
        0,  # first_obs
        float('+inf'),  # min
        float('-inf'),  # max
        0,  # current_count
        0,  # current_obs
        10)  # drop_obs
          )

    Value(StatStreamStruct, 1, 2, 3, 4, 5, 6, 7, 8)
