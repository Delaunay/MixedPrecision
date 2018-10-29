from typing import *

import subprocess
from multiprocessing import Process
from MixedPrecision.tools.stats import StatStream

nvidia_smi = 'nvidia-smi'
metrics = [
    'name',
    'temperature.gpu',
    'utilization.gpu',
    'utilization.memory',
    'memory.total',
    'memory.free',
    'memory.used'
]
query = '--query-gpu=' + ','.join(metrics)


class GpuMonitor:

    def __init__(self, loop_interval, device_id):
        self.options = ['--format=csv', '--loop-ms=' + str(loop_interval), '--id=' + str(device_id)]
        self.streams = [StatStream(drop_first_obs=2) for _ in metrics]
        self.n = len(metrics)
        self.process = None
        self.dispatcher = {
            'name': self.process_ignore,
            'temperature.gpu': self.process_value,
            'utilization.gpu': self.process_percentage,
            'utilization.memory': self.process_percentage,
            'memory.total': self.process_memory,
            'memory.free': self.process_memory,
            'memory.used': self.process_memory
        }

    def run(self):
        with subprocess.Popen([nvidia_smi, query] + self.options, stdout=subprocess.PIPE, bufsize=1) as proc:
            self.process = proc
            for line in proc.stdout.readlines():
                self.parse(line.decode('UTF-8').strip())

    def report(self):
        import MixedPrecision.tools.report as report

        header = ['Metric', 'Average', 'Deviation', 'Min', 'Max', 'count']
        table = []

        for i, stream in enumerate(self.streams):
            table.append([metrics[i]] + stream.to_array())

        report.print_table(header, table)

    def parse(self, line):
        if line == '':
            return

        elems = line.split(',')

        if len(elems) != self.n:
            print('Error line mismatch {} != {} with \n -> `{}`'.format(len(elems), self.n, line))
            return

        for index, value in enumerate(elems):
            self.dispatcher[metrics[index]](index, value)

    def process_percentage(self, index, value):
        try:
            value, _ = value.strip().split(' ')
            self.process_value(index, value)
        except Exception as e:
            print('Expected value format: `66 %` got `{}`'.format(value))
            print(e)

    def process_value(self, index, value):
        self.streams[index] += float(value)

    def process_ignore(self, index, value):
        pass

    def process_memory(self, index, value):
        try:
            value, _ = value.trim.split(' ')
            self.process_value(index, value)
        except Exception as e:
            print('Expected value format: `66 Mib` got `{}`'.format(value))
            print(e)

    def stop(self):
        if self.process is not None:
            self.process.terminate()


def start_monitor(monitor) -> GpuMonitor:
    monitor.run()
    return monitor


def make_monitor(loop_interval=1000, device_id=0) -> Tuple[Process, GpuMonitor]:
    monitor = GpuMonitor(loop_interval, device_id)
    proc = Process(target=start_monitor, args=(monitor,))
    proc.start()
    return proc, monitor


if __name__ == '__main__':
    print('hello')

    proc, mon = make_monitor(loop_interval=1000, device_id=0)
    print(mon)
    pass


