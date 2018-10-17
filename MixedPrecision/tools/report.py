from typing import *


class UnEvenTable(Exception):
    def __init__(self, message):
        self.message = message


class PrintTable:
    """
        Generate a markdown table
    """

    def __init__(self, cols, data):
        self.format = {
            float: '.4f',
            int: '4d'
        }

        self.columns = cols
        self.data = data
        self.col_num = len(self.columns)
        self.check_size()

        self.col_size =self.compute_col_size()
        self.row_size = self.compute_row_size()

    def check_size(self):
        for row_id, row in enumerate(self.data):
            if len(row) != self.col_num:
                raise UnEvenTable('Row ({}) has not the correct number of columns {} != {}'
                                  .format(row_id, len(row), self.col_num))

    def compute_val_size(self, value: Any) -> int:
        return len(self.format_cell(value)) + 2

    def format_cell(self, value: Any) -> str:
        fmt = self.format.get(type(value))
        if fmt is not None:
            return ('{:' + fmt + '}').format(value)
        return str(value)

    def compute_col_size(self):
        col_sizes = [float('-inf')] * self.col_num

        def max_col(_, col_id, val):
            col_sizes[col_id] = max(col_sizes[col_id], self.compute_val_size(val))

        self.foreach(max_col)
        return col_sizes

    def compute_row_size(self):
        return sum(self.col_size)

    def aligned_print(self, str, length, align, end=''):
        missing = max(length - len(str), 0)
        if align == 'left':
            print(str + ' ' * missing + end, end='')

        if align == 'right':
            print(' ' * missing + str + end, end='')

        if align == 'center':
            r = missing // 2
            l = r + missing % 2
            print(' ' * l + str + ' ' * r + end, end='')

        if align is None:
            print(str + end, end='')

    def print(self):
        def simple(rowd_id, col_id, val):
            self.aligned_print(' ' + self.format_cell(val) + ' ', self.col_size[col_id], 'right', end='|')

        def md_header_separator():
            cols = ['-' * (int(size) - 1) + ':' for size in self.col_size]

            print('|' + '|'.join(cols) + '|')

        self.foreach(simple,
                     beg_row=lambda x: print('|', end=''),
                     end_row=lambda x: print(), after_header=md_header_separator)

    def foreach(self, fun, beg_row=None, end_row=None, after_header=None):
        if beg_row is not None:
            beg_row(-1)

        for col_id, header in enumerate(self.columns):
            fun(-1, col_id, header)

        if end_row is not None:
            end_row(-1)

        if after_header is not None:
            after_header()

        for row_id, row in enumerate(self.data):
            if beg_row is not None:
                beg_row(row_id)

            for col_id, col in enumerate(row):
                fun(row_id, col_id, col)

            if end_row is not None:
                end_row(row_id)


def print_table(cols, data):
    PrintTable(cols, data).print()


if __name__ == '__main__':
    PrintTable(['A', 'B', 'C'], [
        ['qwerty', 1.23456789, 4567890],
        ['qwerty', 1.23, 4567890],
        ['qwerty', 4567891.23456789, 4567890]
    ]).print()
