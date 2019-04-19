import torch
import os
import json
import hashlib
import socket

from benchutils.chrono import MultiStageChrono
from benchutils.versioning import get_file_version

from MixedPrecision.tools.containers.ring import RingBuffer
from MixedPrecision.tools.loggers import make_log

from typing import *


excluded_arguments = {
    'report',
    'seed'
}

not_parameter = {
    'gpu'
}


def get_gpu_name():
    import torch
    current_device = torch.cuda.current_device()
    return torch.cuda.get_device_name(current_device)


def get_sync(args):
    import torch

    if hasattr(args, 'cuda') and args.cuda:
        return lambda: torch.cuda.synchronize()

    return lambda: None


def init_torch(args):
    import torch
    import numpy as np

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_cores)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    return torch.device("cuda" if args.cuda else "cpu")


def get_experience_descriptor(name: str) -> Tuple[str, str]:
    version = get_file_version(name)[:10]
    name = '_'.join(name.split('/')[-4:])
    return name, version


class Experiment:
    """ Store all the information we care about during an experiment
            chrono      : Performance timer
            args        : argument passed to the script
            name        : name of the experiment
            batch_loss_buffer : Batch loss values (just making sure we do not get NaNs
            epoch_loss_buffer:  Epoch loss values same as batch loss
    """

    def __init__(self, module, skip_obs=10):
        self.name, self.version = get_experience_descriptor(module)
        self._chrono = None
        self.skip_obs = skip_obs
        self.args = None
        self.batch_loss_buffer = RingBuffer(100, torch.float32)
        self.epoch_loss_buffer = RingBuffer(10, torch.float32)
        self.metrics = {}

        self.remote_logger = None

        try:
            self.remote_logger = make_log(
                api_key=os.environ.get("CML_API_KEY"),
                project_name=self.name,
                workspace=os.environ.get("CML_WORKSPACE")
            )
        except Exception as e:
            print(e)
            self.remote_logger = make_log()

    def get_arguments(self, parser, *args, **kwargs):
        self.args = parser.get_arguments()
        self._chrono = MultiStageChrono(name=self.name, skip_obs=self.skip_obs, sync=get_sync(self.args))

        return self.args

    def chrono(self):
        return self._chrono

    def log_batch_loss(self, val):
        self.batch_loss_buffer.append(val)
        self.remote_logger.log_metric('batch_loss', val)

    def log_epoch_loss(self, val):
        self.epoch_loss_buffer.append(val)
        self.remote_logger.log_metric('epoch_loss', val)

    def log_metric(self, name, val):
        metric = self.metrics.get(name)
        if metric is None:
            metric = RingBuffer(10, torch.float32)
            self.metrics[name] = metric
        metric.append(val)
        self.remote_logger.log_metric(name, val)

    def report(self):
        if self.args is not None:
            args = self.args.__dict__
        else:
            args = {}

        if args['report'] is None:
            args['report'] = os.environ.get('REPORT_PATH')

        filename = args['report']
        # Each GPU has its report we will consolidate later
        filename = f'{filename}_{args["jr_id"]}.json'

        args['version'] = self.version

        unique_id = hashlib.sha256()

        # make it deterministic
        items = list(args.items())
        items.sort()

        for k, w in items:
            if k in not_parameter:
                continue

            unique_id.update(str(k).encode('utf-8'))
            unique_id.update(str(w).encode('utf-8'))

        # we do not want people do modify our shit if the id do not match then they get disqualified
        args['unique_id'] = unique_id.hexdigest()

        # Try to identify vendors so we can find them more easily
        if args['cuda']:
            args['gpu'] = get_gpu_name()

        args['hostname'] = socket.gethostname()
        args['batch_loss'] = self.batch_loss_buffer.to_list()
        args['epoch_loss'] = self.epoch_loss_buffer.to_list()
        args['metrics'] = self.metrics

        for excluded in excluded_arguments:
            args.pop(excluded, None)

        self.remote_logger.log_parameters(args)
        report_dict = self._chrono.to_dict(args)

        # train is the default name for bathed stuff
        if 'train' in report_dict:
            train_data = report_dict['train']

            item_count = report_dict['batch_size'] * report_dict['number']
            min_item = item_count / train_data['max']
            max_item = item_count / train_data['min']

            train_item = {
                'avg': item_count / train_data['avg'],
                'max': max_item,
                'min': min_item,
                'range': max_item - min_item,
                'unit': 'items/sec'
            }

            report_dict['train_item'] = train_item

        print('-' * 80)
        json_report = json.dumps(report_dict, sort_keys=True, indent=4, separators=(',', ': '))
        print(json_report)

        if not os.path.exists(filename):
            report_file = open(filename, 'w')
            report_file.write('[')
            report_file.close()

        report_file = open(filename, 'a')
        report_file.write(json_report)
        report_file.write(',')
        report_file.close()

    print('-' * 80)

    def get_device(self):
        return init_torch(self.args)

    def get_time(self, stream):
        if stream.avg == 0:
            return stream.val
        return stream.avg

    def show_eta(self, epoch_id, timer, msg=''):
        if msg:
            msg = ' | ' + msg

        loss = self.batch_loss_buffer.last()
        if loss is not None:
            loss = f'| Batch Loss {loss:8.4f}'
        else:
            loss = ''

        print(
            f'[{epoch_id:3d}/{self.args.repeat:3d}] '
            f'| ETA: {self.get_time(timer) * (self.args.repeat - (epoch_id + 1)) / 60:6.2f} min ' + loss + msg
        )
