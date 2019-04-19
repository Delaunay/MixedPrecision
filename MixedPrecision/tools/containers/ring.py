import array
import torch


class RingBuffer:
    types = {
        torch.float16: 'f',  # 4
        torch.float32: 'f',  # 4
        torch.float64: 'd',  # 8

        torch.int8:  'b',    # 1
        torch.int16: 'h',    # 2
        torch.int32: 'l',    # 4
        torch.int64: 'q',    # 8

        torch.uint8:  'B',   # 1
        # torch.uint16: 'H',   # 2
        # torch.uint32: 'L',   # 4
        # torch.uint64: 'Q',   # 8
    }

    def __init__(self, size, dtype, default_val=0):
        self.array = array.array(self.types[dtype], [default_val] * size)
        self.capacity = size
        self.offset = 0

    def __getitem__(self, item):
        return self.array[item % self.capacity]

    def __setitem__(self, item, value):
        self.array[item % self.capacity] = value

    def append(self, item):
        self.array[self.offset % self.capacity] = item
        self.offset += 1

    def to_list(self):
        if self.offset < self.capacity:
            return list(self.array[:self.offset])
        else:
            end_idx = self.offset % self.capacity
            return list(self.array[end_idx: self.capacity]) + list(self.array[0:end_idx])

    def __len__(self):
        return min(self.capacity, self.offset)

    def last(self):
        if self.offset == 0:
            return None

        return self.array[(self.offset - 1) % self.capacity]
