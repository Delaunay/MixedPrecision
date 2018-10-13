import math


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
        self.reset()
        self.drop_obs = drop_first_obs

    def reset(self):
        self.sum = 0.0
        self.sum_sqr = 0.0
        self.current_count = 0
        self.current_obs = 0
        self.first_obs = 0
        self.min = float('inf')
        self.max = float('-inf')

    def __iadd__(self, other):
        self.update(other, 1)
        return self

    def update(self, val, weight=1):
        self.current_count += weight

        if self.current_count < self.drop_obs:
            self.current_obs = val
            return

        if self.count == 1:
            self.first_obs = val

        self.current_obs = val - self.first_obs
        self.sum += float(self.current_obs) * float(weight)
        self.sum_sqr += float(self.current_obs * self.current_obs) * float(weight)

        self.min = min(self.min, val)
        self.max = max(self.max, val)

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
