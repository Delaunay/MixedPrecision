from typing import *

import subprocess
from multiprocessing import Process
from MixedPrecision.tools.stats import StatStream
from collections import OrderedDict
import time

metrics = [
    'memory.used',
    'memory.free',
    'memory.total',
    'temperature.gpu',
    'utilization.gpu',
    'utilization.memory',
]


class AmdGpuMonitor:
    """ ROCm smi utility is too basic for us to use.
        We will use sysfs/KFD api instead.

        One problem with KFD is that the node number is not guaranteed to be the same after reboot
        which sucks. I only have one GPU so I do not care.
    """
    def __init__(self, loop_interval, device_id):
        self.streams = [StatStream(drop_first_obs=2) for _ in metrics]
        self.n = len(metrics)
        self.running = True
        self.sleep_time=loop_interval

        # this requires sudo
        self.powerprefix = '/sys/kernel/debug/dri/'

        self.drmprefix = '/sys/class/drm'
        self.hwmonprefix = '/sys/class/hwmon'
        self.kfdprefix = '/sys/devices/virtual/kfd'

        self.file_names = {
            'memory': self.kfdprefix + '/kfd/topology/nodes/1/mem_banks/0/used_memory',
            'memory_property': self.kfdprefix + '/kfd/topology/nodes/1/mem_banks/0/properties',
            'gpu_usage': self.hwmonprefix + '/hwmon0/device/gpu_busy_percent',
            'temperature': self.hwmonprefix + '/hwmon0/temp1_input',
        }
        # opened file cache
        self.files = {}

        self.total_memory = self.parse_memory_props()
        self.used_memory = None

        self.metrics = OrderedDict({
            # 'name': self.parse_name,
            'temperature.gpu': self.parse_temperature,
            'utilization.gpu': self.parse_gpu_usage,
            'utilization.memory': self.parse_memory_usage,
            'memory.total': self.parse_memory_total,
            'memory.free': self.parse_memory_free,
            'memory.used': self.parse_memory_used
        })

    def read_props(self, file):
        props = {}

        with open(file, 'r') as f:
            for line in f.readlines():
                key, val = line.split(' ')
                props[key] = val

        return props

    def parse_memory_props(self):
        return int(self.read_props(self.file_names['memory_property'])['size_in_bytes']) / (1024 * 1024)

    def read_value(self, file_name):
        file = self.files.get(file_name)

        if file is None:
            file = open(self.file_names[file_name], 'r')

        file.seek(0)
        temp = file.readline()
        return temp

    def parse_temperature(self):
        return int(self.read_value('temperature'))

    def parse_gpu_usage(self):
        return float(self.read_value('gpu_usage'))

    def parse_memory_used(self):
        self.used_memory = int(self.read_value('memory')) / (1024 * 1024)
        return self.used_memory

    def parse_memory_usage(self):
        return self.used_memory * 100.0 / self.total_memory

    def parse_memory_total(self):
        return self.total_memory

    def parse_memory_free(self):
        return self.total_memory - self.used_memory

    def parse_name(self):
        return ''

    def run(self):
        print('Running ROCm Monitor')
        while self.running:
            for i, metric in enumerate(metrics):
                self.streams[i].update(self.metrics[metric]())

            time.sleep(self.sleep_time / 1000)

    def stop(self):
        self.running = False

    def report(self):
        import MixedPrecision.tools.report as report

        header = ['Metric', 'Average', 'Deviation', 'Min', 'Max', 'count']
        table = []

        for i, stream in enumerate(self.streams):
            table.append([metrics[i]] + stream.to_array())

        report.print_table(header, table)

    def arrays(self, common):
        return [['gpu.' + metrics[i]] + stream.to_array() + common for i, stream in enumerate(self.streams)]


if __name__ == '__main__':

    mon = AmdGpuMonitor(250, 0)

    mon.run()
